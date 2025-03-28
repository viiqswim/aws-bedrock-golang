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
	// The template for the series name extraction prompt
	promptTemplate = `You are a series name extraction tool that ONLY outputs valid JSON.

INPUT: "%s"

INSTRUCTIONS:
1. Extract ONLY the series name (text that appears before "Season" or "Episode")
2. Return ONLY a valid JSON array with format: [{"series": "extracted name"}]
3. DO NOT include any explanation, additional examples, or commentary
4. The response must contain NOTHING except the JSON array

For example, from "Friends Season 1", extract just "Friends" and output [{"series": "Friends"}]`
)

func main() {
	// Define command-line flags
	modelFlag := flag.String("model", "nova", "The LLM model to use: 'nova', 'llama', 'llama70b', 'claude', or 'deepseek'")
	inputSeriesNameFlag := flag.String("input", "", "The input series name to extract")

	// Parse command-line flags
	flag.Parse()

	// Convert model name to lowercase for case-insensitive comparison
	modelName := strings.ToLower(*modelFlag)
	inputSeriesName := *inputSeriesNameFlag
	if inputSeriesName == "" {
		inputSeriesName = "Friends Season 001 Episode 001"
	}

	// Format the prompt with the input series name
	prompt := fmt.Sprintf(promptTemplate, inputSeriesName)

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

	if inputSeriesName == "" {
		log.Fatalf("Input series name cannot be empty. Provide a valid input using the -input flag.")
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
