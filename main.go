package main

import (
	"bedrock-llama/claude"
	"bedrock-llama/deepseek"
	"bedrock-llama/llama"
	"bedrock-llama/llama70b"
	"bedrock-llama/nova"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/joho/godotenv"
)

const (
	// The default prompt to send to the models
	defaultPrompt = "Hello, how are you"
)

func main() {
	// Define command-line flags
	modelFlag := flag.String("model", "nova", "The LLM model to use: 'nova', 'llama', 'llama70b', 'claude', or 'deepseek'")
	promptFlag := flag.String("prompt", defaultPrompt, "The prompt to send to the model")

	// Parse command-line flags
	flag.Parse()

	// Convert model name to lowercase for case-insensitive comparison
	modelName := strings.ToLower(*modelFlag)
	prompt := *promptFlag

	// Validate model selection
	validModels := map[string]bool{
		"nova":     true,
		"llama":    true,
		"llama70b": true,
		"claude":   true,
		"deepseek": true,
	}

	if !validModels[modelName] {
		log.Fatalf("Invalid model specified. Use 'nova', 'llama', 'llama70b', 'claude', or 'deepseek'")
	}

	fmt.Println("Loading environment variables...")
	if err := godotenv.Load(); err != nil {
		log.Printf("Warning: Error loading .env file: %v", err)
	}

	accessKeyId := os.Getenv("AWS_ACCESS_KEY_ID")
	secretAccessKey := os.Getenv("AWS_SECRET_ACCESS_KEY")
	awsRegion := os.Getenv("AWS_REGION")

	// Additional diagnostic information
	log.Printf("Using AWS region: %s", awsRegion)
	log.Printf("AWS access key ID present: %v", accessKeyId != "")
	log.Printf("AWS secret access key present: %v", secretAccessKey != "")

	if accessKeyId == "" || secretAccessKey == "" || awsRegion == "" {
		log.Fatalf("Missing required environment variables: AWS_ACCESS_KEY_ID and/or AWS_SECRET_ACCESS_KEY and/or AWS_REGION")
	}

	ctx := context.Background()

	switch modelName {
	case "nova":
		// Run Nova model
		fmt.Println("Invoking Amazon Bedrock Nova model...")
		fmt.Printf("Prompt: %s\n", prompt)
		response, err := nova.InvokeModel(ctx, prompt, accessKeyId, secretAccessKey, awsRegion)
		if err != nil {
			log.Fatalf("Error: %v", err)
		}
		nova.PrintResponse(response)
	case "llama":
		// Run Llama model
		fmt.Println("Invoking Amazon Bedrock Llama model...")
		response, err := llama.InvokeModel(ctx, prompt, accessKeyId, secretAccessKey, awsRegion)
		if err != nil {
			log.Fatalf("Error: %v", err)
		}
		llama.PrintResponse(response)
	case "llama70b":
		// Run Llama 3.3 70B model
		fmt.Println("Invoking Amazon Bedrock Llama 3.3 70B model...")
		fmt.Printf("Prompt: %s\n", prompt)
		response, err := llama70b.InvokeModel(ctx, prompt, accessKeyId, secretAccessKey, awsRegion)
		if err != nil {
			log.Fatalf("Error: %v", err)
		}
		llama70b.PrintResponse(response)
	case "claude":
		// Run Claude model
		fmt.Println("Invoking Amazon Bedrock Claude 3 Sonnet model...")
		fmt.Printf("Prompt: %s\n", prompt)
		response, err := claude.InvokeModel(ctx, prompt, accessKeyId, secretAccessKey, awsRegion)
		if err != nil {
			log.Fatalf("Error: %v", err)
		}
		claude.PrintResponse(response)
	case "deepseek":
		// Run DeepSeek model
		fmt.Println("Invoking Amazon Bedrock DeepSeek model...")
		fmt.Printf("Prompt: %s\n", prompt)
		response, err := deepseek.InvokeModel(ctx, prompt, accessKeyId, secretAccessKey, awsRegion)
		if err != nil {
			log.Fatalf("Error: %v", err)
		}
		deepseek.PrintResponse(response)
	}
}
