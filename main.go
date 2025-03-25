package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"

	"github.com/aws/aws-sdk-go-v2/aws"
	"github.com/aws/aws-sdk-go-v2/config"
	"github.com/aws/aws-sdk-go-v2/credentials"
	"github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	"github.com/joho/godotenv"
)

const (
	// Use the AWS Bedrock inference profile ARN
	modelID = "arn:aws:bedrock:us-east-2:913524932967:inference-profile/us.meta.llama3-2-1b-instruct-v1:0"

	// The prompt to send to the model
	prompt = "Hello, how are you"
)

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

func main() {
	fmt.Println("Invoking Amazon Bedrock model...")

	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: Error loading .env file: %v", err)
	}

	ctx := context.Background()

	accessKeyId := os.Getenv("AWS_ACCESS_KEY_ID")
	secretAccessKey := os.Getenv("AWS_SECRET_ACCESS_KEY")
	awsRegion := os.Getenv("AWS_REGION")

	if accessKeyId == "" || secretAccessKey == "" || awsRegion == "" {
		log.Fatalf("Missing required environment variables: AWS_ACCESS_KEY_ID and/or AWS_SECRET_ACCESS_KEY and/or AWS_REGION")
	}

	cfg, err := config.LoadDefaultConfig(ctx,
		config.WithRegion(awsRegion),
		config.WithCredentialsProvider(credentials.NewStaticCredentialsProvider(
			accessKeyId,
			secretAccessKey,
			"", // Session token (empty for regular access keys)
		)),
	)
	if err != nil {
		log.Fatalf("Failed to load AWS configuration: %v", err)
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
		log.Fatalf("Failed to marshal payload: %v", err)
	}

	// Create the input for the InvokeModel operation
	input := &bedrockruntime.InvokeModelInput{
		ModelId:     aws.String(modelID),
		ContentType: aws.String("application/json"),
		Body:        payloadBytes,
	}
	// Invoke the model
	output, err := client.InvokeModel(ctx, input)
	if err != nil {
		log.Fatalf("Error invoking Bedrock model: %v", err)
	}

	var response Response
	if err := json.Unmarshal(output.Body, &response); err != nil {
		log.Fatalf("Failed to unmarshal response: %v", err)
	}

	fmt.Printf("Response: %s\n", response.Generation)

	// Print token usage information if available
	if response.Usage.InputTokens > 0 || response.Usage.OutputTokens > 0 {
		fmt.Printf("Input tokens: %d\n", response.Usage.InputTokens)
		fmt.Printf("Output tokens: %d\n", response.Usage.OutputTokens)
	}
}
