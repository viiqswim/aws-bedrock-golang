package claude

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

// ModelID is the AWS Bedrock Claude 3 Sonnet model ID
const ModelID = "arn:aws:bedrock:us-east-2:913524932967:inference-profile/us.anthropic.claude-3-5-sonnet-20241022-v2:0"

// ContentItem represents a content item in the message
type ContentItem struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

// Message represents a message in the conversation
type Message struct {
	Role    string        `json:"role"`
	Content []ContentItem `json:"content"`
}

// Payload represents the request payload for the Claude model
type Payload struct {
	AnthropicVersion string    `json:"anthropic_version"`
	MaxTokens        int       `json:"max_tokens"`
	TopK             int       `json:"top_k"`
	StopSequences    []string  `json:"stop_sequences"`
	Temperature      float64   `json:"temperature"`
	TopP             float64   `json:"top_p"`
	Messages         []Message `json:"messages"`
}

// Response represents the response from the Claude model
type Response struct {
	ID      string `json:"id"`
	Type    string `json:"type"`
	Role    string `json:"role"`
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Model        string      `json:"model"`
	StopReason   string      `json:"stop_reason"`
	StopSequence interface{} `json:"stop_sequence"`
	Usage        struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// InvokeModel calls the Claude model with the given prompt
func InvokeModel(ctx context.Context, prompt string, accessKeyId, secretAccessKey, awsRegion string) (*Response, error) {
	// Debug output to verify prompt
	log.Printf("Sending prompt to Claude model: %s", prompt)

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

	// Prepare payload according to Claude requirements
	payload := Payload{
		AnthropicVersion: "bedrock-2023-05-31",
		MaxTokens:        200,
		TopK:             250,
		StopSequences:    []string{},
		Temperature:      1.0,
		TopP:             0.999,
		Messages: []Message{
			{
				Role: "user",
				Content: []ContentItem{
					{
						Type: "text",
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
	log.Printf("Claude payload: %s", string(payloadBytes))

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
		return nil, fmt.Errorf("error invoking Bedrock Claude model: %v", err)
	}

	// Debug: Log the raw response
	log.Printf("Raw Claude response: %s", string(output.Body))

	var response Response
	if err := json.Unmarshal(output.Body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Claude response: %v", err)
	}

	// Debug: Log the parsed response structure
	responseBytes, _ := json.MarshalIndent(response, "", "  ")
	log.Printf("Parsed Claude response: %s", string(responseBytes))

	return &response, nil
}

// PrintResponse formats and prints the Claude model response
func PrintResponse(response *Response) {
	// Extract and print the response text
	if len(response.Content) > 0 {
		fmt.Printf("Response: %s\n", response.Content[0].Text)
	} else {
		fmt.Println("No response content received from Claude model")
		fmt.Printf("Response structure: %+v\n", response)
	}

	// Print token usage information if available
	fmt.Printf("Input tokens: %d\n", response.Usage.InputTokens)
	fmt.Printf("Output tokens: %d\n", response.Usage.OutputTokens)
}
