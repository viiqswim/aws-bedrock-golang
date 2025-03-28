# LLM Series Name Extractor (Go)

This project provides a simple Go interface for interacting with AWS Bedrock LLM models (Llama, Llama 3.3 70B, Nova, Claude, and DeepSeek).

## Project Overview

This client allows you to:

- Connect to AWS Bedrock service
- Send prompts to AWS Bedrock models
- Process and display model responses
- Track token usage statistics

## Prerequisites

- Go 1.16 or higher
- AWS account with access to AWS Bedrock
- Appropriate IAM permissions to invoke Bedrock models

## Setup

1. Clone the repository:

   ```
   git clone <repository-url>
   cd llm-series-name-extractor-go
   ```

2. Install dependencies:

   ```
   go mod tidy
   ```

3. Set up environment variables by creating a `.env` file:

   ```
   AWS_ACCESS_KEY_ID=your_aws_access_key
   AWS_SECRET_ACCESS_KEY=your_aws_secret_key
   AWS_REGION=us-east-2
   ```

## AWS Configuration

### Required AWS Setup

1. **AWS Credentials**: Configure using one of these methods:

- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN`
- Shared credentials file: `~/.aws/credentials`
- **Using a .env file** (recommended for development):
  - Create a .env file in the project root
  - Add your AWS credentials and region:

    ```text
    AWS_ACCESS_KEY_ID=your_access_key_here
    AWS_SECRET_ACCESS_KEY=your_secret_key_here
    AWS_REGION=us-east-2
    ```

  - **Important**: Make sure the region in your .env file matches the region in the modelID ARN (e.g. us-east-2)
  - The application automatically loads this file using the godotenv package

2. **Amazon Bedrock Access**:

- Ensure your AWS account has Bedrock service enabled
- Request and approve model access for Meta Llama 3 in the AWS Bedrock console
- Create an inference profile and use the ARN in the code in place of the model ID

3. **IAM Permissions**:
Ensure your IAM role or user has the following permissions:

- `bedrock:InvokeModel`
- `bedrock:ListFoundationModels`
- `bedrock:GetFoundationModel`

## Usage

### Command-line Arguments

The application supports the following command-line arguments:

#### Selecting the LLM Model

You can specify which model to use with the `-model` flag:

```bash
# To use the Nova model (default)
go run main.go -model=nova

# To use the Llama 3.2 1B model
go run main.go -model=llama

# To use the Llama 3.3 70B model
go run main.go -model=llama70b

# To use the Claude 3 Sonnet model
go run main.go -model=claude

# To use the DeepSeek model
go run main.go -model=deepseek
```

#### Customizing the Prompt

You can provide a custom prompt with the `-prompt` flag:

```bash
go run main.go -prompt="Explain quantum computing in simple terms"
```

#### Combining Options

You can combine both options:

```bash
go run main.go -model=llama -prompt="What are the benefits of GraphQL over REST?"
```

### Basic Usage

Run the application with default settings (uses Nova model):

```bash
go run main.go
```

By default, the application will:

1. Connect to AWS Bedrock using your credentials
2. Send the default prompt to the selected model
3. Display the model's response
4. Show token usage information if available

### Examples

#### Example 1: Ask Nova about a topic

```bash
go run main.go -model=nova -prompt="What are the key features of Go programming language?"
```

#### Example 2: Use Llama for creative writing

```bash
go run main.go -model=llama -prompt="Write a short poem about programming"
```

#### Example 3: Use Claude for complex reasoning

```bash
go run main.go -model=claude -prompt="Explain the pros and cons of microservices architecture"
```

#### Example 4: Use DeepSeek for code generation

```bash
go run main.go -model=deepseek -prompt="Write a function in Go that checks if a string is a palindrome"
```

#### Example 5: Use Llama 3.3 70B for complex reasoning

```bash
go run main.go -model=llama70b -prompt="Compare and contrast different approaches to natural language processing"
```

### Configuration Options

You can customize the behavior by modifying the following constants in `main.go`:

```go
const (
    // Default prompt to send to the model if none provided
    defaultPrompt = "Hello, how are you"
)
```

### Model Parameters

The following parameters can be adjusted in each model's implementation:

- For Llama 3.2 1B:
  - `MaxGenLen`: Maximum length of the generated response (default: 512)
  - `Temperature`: Controls randomness in the output (default: 0.7)
  - `TopP`: Controls diversity via nucleus sampling (default: 0.9)
  
- For Llama 3.3 70B:
  - `MaxGenLen`: Maximum length of the generated response (default: 512)
  - `Temperature`: Controls randomness in the output (default: 0.5)
  - `TopP`: Controls diversity via nucleus sampling (default: 0.9)
  
- For Nova:
  - `MaxNewTokens`: Maximum number of tokens to generate (default: 512)
  - `Temperature`: Controls randomness in the output (default: 0.7)
  - `TopP`: Controls diversity via nucleus sampling (default: 0.9)
  
- For Claude:
  - `MaxTokens`: Maximum tokens to generate (default: 200)
  - `TopK`: Number of tokens to consider for sampling (default: 250)
  - `Temperature`: Controls randomness (default: 1.0)
  - `TopP`: Controls diversity via nucleus sampling (default: 0.999)

- For DeepSeek:
  - `MaxTokens`: Maximum tokens to generate (default: 512)

## Error Handling

The application includes error handling for:

- AWS configuration and authentication issues
- JSON marshaling/unmarshaling errors
- API request failures
- Model invocation errors

## Troubleshooting

If you encounter errors, check:

- AWS credentials are correctly configured
- You have access to the specified model
- The model ID or ARN is correct
- Your region matches the one in the inference profile ARN
