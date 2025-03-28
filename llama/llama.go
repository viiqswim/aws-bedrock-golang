package llama

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// ModelID is the AWS Bedrock inference profile ARN for Llama
const ModelID = "arn:aws:bedrock:us-east-2:913524932967:inference-profile/us.meta.llama3-2-1b-instruct-v1:0"

// Payload represents the request payload for the Meta Llama model
type Payload struct {
	Prompt      string  `json:"prompt"`
	MaxGenLen   int     `json:"max_gen_len"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
}

// Response represents the response from the Meta Llama model
type Response struct {
	Generation string `json:"generation"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// InvokeModel calls the Llama model with the given prompt
func InvokeModel(ctx context.Context, prompt string, accessKeyId, secretAccessKey, awsRegion string) (*Response, error) {
	cfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(awsRegion),
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
			accessKeyId,
			secretAccessKey,
			"", // Session token (empty for regular access keys)
		)),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to load AWS configuration: %v", err)
	}

	// Create a Bedrock Runtime client
	client := bedrockruntime.NewFromConfig(cfg)

	// Prepare payload according to Meta Llama requirements
	payload := Payload{
		Prompt:      prompt,
		MaxGenLen:   512,
		Temperature: 0.7,
		TopP:        0.9,
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %v", err)
	}

	// Create the input for the InvokeModel operation
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(ModelID),
		ContentType: aws.String("application/json"),
		Body:        payloadBytes,
	}

	// Invoke the model
	output, err := client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("error invoking Bedrock model: %v", err)
	}

	var response Response
	if err := json.Unmarshal(output.Body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	return &response, nil
}

// PrintResponse formats and prints the Llama model response
func PrintResponse(response *Response) {
	fmt.Printf("Response: %s\n", response.Generation)

	// Print token usage information if available
	if response.Usage.InputTokens > 0 || response.Usage.OutputTokens > 0 {
		fmt.Printf("Input tokens: %d\n", response.Usage.InputTokens)
		fmt.Printf("Output tokens: %d\n", response.Usage.OutputTokens)
	}
}
