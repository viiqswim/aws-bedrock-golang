package nova

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// ModelID is the AWS Bedrock Nova inference profile ARN
const ModelID = "arn:aws:bedrock:us-east-2:913524932967:inference-profile/us.amazon.nova-pro-v1:0"

// Content represents a message content item
type Content struct {
	Text string `json:"text"`
}

// Message represents a message in the conversation
type Message struct {
	Role    string    `json:"role"`
	Content []Content `json:"content"`
}

// InferenceConfig represents the configuration for the inference
type InferenceConfig struct {
	MaxNewTokens int     `json:"max_new_tokens"`
	Temperature  float64 `json:"temperature"`
	TopP         float64 `json:"top_p"`
}

// Payload represents the request payload for the Amazon Nova model
type Payload struct {
	InferenceConfig InferenceConfig `json:"inferenceConfig"`
	Messages        []Message       `json:"messages"`
}

// Response represents the response from the Amazon Nova model
type Response struct {
	Output struct {
		Role    string    `json:"role"`
		Content []Content `json:"content"`
		Stop    bool      `json:"stop"`
	} `json:"output"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// InvokeModel calls the Nova model with the given prompt
func InvokeModel(ctx context.Context, prompt string, accessKeyId, secretAccessKey, awsRegion string) (*Response, error) {
	// Debug output to verify prompt
	log.Printf("Sending prompt to Nova model: %s", prompt)

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

	// Prepare payload according to Amazon Nova requirements
	payload := Payload{
		InferenceConfig: InferenceConfig{
			MaxNewTokens: 512,
			Temperature:  0.7,
			TopP:         0.9,
		},
		Messages: []Message{
			{
				Role: "user",
				Content: []Content{
					{
						Text: prompt,
					},
				},
			},
		},
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %v", err)
	}

	// Debug: Log the payload being sent to the model
	log.Printf("Payload: %s", string(payloadBytes))

	// Create the input for the InvokeModel operation
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(ModelID),
		ContentType: aws.String("application/json"),
		Body:        payloadBytes,
	}

	// Invoke the model
	output, err := client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("error invoking Bedrock Nova model: %v", err)
	}

	// Debug: Log the raw response
	log.Printf("Raw response: %s", string(output.Body))

	var response Response
	if err := json.Unmarshal(output.Body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal response: %v", err)
	}

	// Debug: Log the parsed response structure
	responseBytes, _ := json.MarshalIndent(response, "", "  ")
	log.Printf("Parsed response: %s", string(responseBytes))

	return &response, nil
}

// PrintResponse formats and prints the Nova model response
func PrintResponse(response *Response) {
	// Extract and print the response text
	if len(response.Output.Content) > 0 {
		fmt.Printf("Response: %s\n", response.Output.Content[0].Text)
	} else {
		fmt.Println("No response content received. This may indicate an issue with the model configuration.")
		fmt.Println("Please check:\n- Your AWS credentials\n- The model ARN\n- The AWS region")
		fmt.Printf("Response structure: %+v\n", response)
	}

	// Print token usage information if available
	fmt.Printf("Input tokens: %d\n", response.Usage.InputTokens)
	fmt.Printf("Output tokens: %d\n", response.Usage.OutputTokens)
}
