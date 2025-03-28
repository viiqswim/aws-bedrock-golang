package llama70b

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
)

// ModelID is the AWS Bedrock inference profile ARN for Llama 3.3 70B
const ModelID = "arn:aws:bedrock:us-east-2:913524932967:inference-profile/us.meta.llama3-3-70b-instruct-v1:0"

// Payload represents the request payload for the Meta Llama 3.3 70B model
// Note: The structure is the same as Llama 3.2, but we use different default values
type Payload struct {
	Prompt      string  `json:"prompt"`
	MaxGenLen   int     `json:"max_gen_len"`
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
}

// Response represents the response from the Meta Llama 3.3 70B model
type Response struct {
	Generation string `json:"generation"`
	Usage      struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
}

// InvokeModel calls the Llama 3.3 70B model with the given prompt
func InvokeModel(ctx context.Context, prompt string, accessKeyId, secretAccessKey, awsRegion string) (*Response, error) {
	// Debug output to verify prompt
	log.Printf("=== PROMPT ===\n%s\n============", prompt)

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

	// Prepare payload according to Meta Llama 3.3 70B requirements
	// Using recommended settings for the 70B model with lower temperature
	payload := Payload{
		Prompt:      prompt,
		MaxGenLen:   64,   // Reduced from 128 to further limit output
		Temperature: 0.01, // Further reduced to make output more deterministic
		TopP:        0.5,  // Reduced to focus on the most likely tokens
	}

	payloadBytes, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal payload: %v", err)
	}

	// Debug: Log the payload being sent to the model
	log.Printf("=== PAYLOAD ===\n%s\n=============", string(payloadBytes))

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
		return nil, fmt.Errorf("error invoking Bedrock Llama 3.3 70B model: %v", err)
	}

	// Debug: Log the raw response
	log.Printf("=== RAW RESPONSE ===\n%s\n==================", string(output.Body))

	var response Response
	if err := json.Unmarshal(output.Body, &response); err != nil {
		return nil, fmt.Errorf("failed to unmarshal Llama 3.3 70B response: %v", err)
	}

	// Debug: Log the parsed response structure
	responseBytes, _ := json.MarshalIndent(response, "", "  ")
	log.Printf("=== PARSED RESPONSE ===\n%s\n=====================", string(responseBytes))

	return &response, nil
}

// PrintResponse formats and prints the Llama 3.3 70B model response
func PrintResponse(response *Response) {
	output := response.Generation

	// Try to find the JSON array pattern and extract it
	jsonPattern := regexp.MustCompile(`\[\s*{\s*"series"\s*:\s*"([^"]*)"\s*}\s*\]`)
	if match := jsonPattern.FindStringSubmatch(output); len(match) > 1 {
		fmt.Printf("[{\"series\": \"%s\"}]\n", match[1])
	} else {
		// Try a fallback approach to extract just the series name
		seriesPattern := regexp.MustCompile(`"series"\s*:\s*"([^"]*)"`)
		if match := seriesPattern.FindStringSubmatch(output); len(match) > 1 {
			fmt.Printf("[{\"series\": \"%s\"}]\n", match[1])
		} else {
			// Last resort: print the cleaned response
			fmt.Println(strings.TrimSpace(output))
		}
	}

	// Print token usage information as logs
	log.Printf("Input tokens: %d\n", response.Usage.InputTokens)
	log.Printf("Output tokens: %d\n", response.Usage.OutputTokens)
}
