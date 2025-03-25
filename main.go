	package main

	import (
	    "context"
	    "encoding/json"
	    "fmt"
	    "log"

	    "github.com/aws/aws-sdk-go-v2/aws"
	    "github.com/aws/aws-sdk-go-v2/config"
	    "github.com/aws/aws-sdk-go-v2/service/bedrockruntime"
	)

	// Model and region configuration
	const (
	    // Match the region with the one in the inference profile ARN
	    awsRegion = "us-east-2"

	    // Use the correct inference profile ARN
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

	    // Create a context
	    ctx := context.Background()

	    // Load AWS configuration with the specified region
	    cfg, err := config.LoadDefaultConfig(ctx, config.WithRegion(awsRegion))
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

	    // Convert payload to JSON
	    payloadBytes, err := json.Marshal(payload)
	    if err != nil {
	        log.Fatalf("Failed to marshal payload: %v", err)
	    }

	    // Invoke the model
	    input := &bedrockruntime.InvokeModelInput{
	        ModelId:     aws.String(modelID),
	        ContentType: aws.String("application/json"),
	        Body:        payloadBytes,
	    }

	    output, err := client.InvokeModel(ctx, input)
	    if err != nil {
	        log.Fatalf("Error invoking Bedrock model: %v", err)
	    }

	    // Parse the response
	    var response Response
	    if err := json.Unmarshal(output.Body, &response); err != nil {
	        log.Fatalf("Failed to unmarshal response: %v", err)
	    }

	    // Print the response
	    fmt.Printf("Response: %s\n", response.Generation)

	    // Print token usage information if available
	    if response.Usage.InputTokens > 0 || response.Usage.OutputTokens > 0 {
	        fmt.Printf("Input tokens: %d\n", response.Usage.InputTokens)
	        fmt.Printf("Output tokens: %d\n", response.Usage.OutputTokens)
	    }
	}

