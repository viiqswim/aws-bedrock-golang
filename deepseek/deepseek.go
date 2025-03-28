package deepseek

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

// ModelID is the AWS Bedrock DeepSeek inference profile ARN
const ModelID = "arn:aws:bedrock:us-east-2:913524932967:inference-profile/us.deepseek.r1-v1:0"

// Message represents a message in the conversation
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// InferenceConfig represents the configuration for the inference
type InferenceConfig struct {
	MaxTokens int `json:"max_tokens"`
}

// Payload represents the request payload for the DeepSeek model
type Payload struct {
	InferenceConfig InferenceConfig `json:"inferenceConfig"`
	Messages        []Message       `json:"messages"`
}

// Response represents the response from the DeepSeek model
type Response struct {
	Choices []struct {
		Message struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// InvokeModel calls the DeepSeek model with the given prompt
func InvokeModel(ctx context.Context, prompt string, accessKeyId, secretAccessKey, awsRegion string) (*Response, error) {
	// Debug output to verify prompt
	log.Printf("Sending prompt to DeepSeek model: %s", prompt)

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

	// Prepare payload according to DeepSeek requirements
	payload := Payload{
		InferenceConfig: InferenceConfig{
			MaxTokens: 512,
		},
		Messages: []Message{
			{
				Role:    "user",
				Content: prompt,
			},
		},
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %v", err)
	}

	// Debug: Log the payload being sent to the model
	log.Printf("DeepSeek payload: %s", string(payloadBytes))

	// Create the input for the InvokeModel operation
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(ModelID),
		ContentType: aws.String("application/json"),
		Accept:      aws.String("application/json"),
		Body:        payloadBytes,
	}

	// Invoke the model
	output, err := client.InvokeModel(ctx, input)
	if err != nil {
		return nil, fmt.Errorf("error invoking Bedrock DeepSeek model: %v", err)
	}

	// Debug: Log the raw response
	log.Printf("Raw DeepSeek response: %s", string(output.Body))

	var response Response
	if err := json.Unmarshal(output.Body, &response); err != nil {
		// Try to unmarshal into a map to see the actual response structure
		var rawResponse map[string]interface{}
		if jsonErr := json.Unmarshal(output.Body, &rawResponse); jsonErr == nil {
			rawBytes, _ := json.MarshalIndent(rawResponse, "", "  ")
			log.Printf("Raw response structure: %s", string(rawBytes))
		}

		return nil, fmt.Errorf("failed to unmarshal DeepSeek response: %v", err)
	}

	// Debug: Log the parsed response structure
	responseBytes, _ := json.MarshalIndent(response, "", "  ")
	log.Printf("Parsed DeepSeek response: %s", string(responseBytes))

	return &response, nil
}

// PrintResponse formats and prints the DeepSeek model response
func PrintResponse(response *Response) {
	// Extract and print the response text
	if len(response.Choices) > 0 {
		fmt.Printf("Response: %s\n", response.Choices[0].Message.Content)
	} else {
		fmt.Println("No response content received from DeepSeek model")
		fmt.Printf("Response structure: %+v\n", response)
	}

	// Print token usage information if available
	fmt.Printf("Input tokens: %d\n", response.Usage.InputTokens)
	fmt.Printf("Output tokens: %d\n", response.Usage.OutputTokens)
}
