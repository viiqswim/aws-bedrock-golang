# AWS Bedrock Llama 3 Go Client

This project provides a Go client for interacting with Meta Llama 3 models through Amazon Bedrock. It demonstrates how to use the AWS SDK for Go to send prompts to Llama 3 and process the responses.

## Project Overview

This client allows you to:

- Connect to AWS Bedrock service
- Send prompts to Meta Llama 3 models
- Process and display model responses
- Track token usage statistics

## Prerequisites

- Go 1.18 or later
- An AWS account with Amazon Bedrock access
- Permissions to use Meta Llama 3 models in Bedrock
- AWS credentials configured on your system

## Installation

1. Clone the repository:

2. Install the required dependencies:

```bash
go mod tidy
```

## AWS Configuration

### Required AWS Setup

1. **AWS Credentials**: Configure using one of these methods:

- AWS CLI: `aws configure`
- Environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, and optionally `AWS_SESSION_TOKEN`
- Shared credentials file: `~/.aws/credentials`

2. **Amazon Bedrock Access**:

- Ensure your AWS account has Bedrock service enabled
- Request and approve model access for Meta Llama 3 in the AWS Bedrock console
- Create an inference profile and use the ARN in the code in place of the model ID

1. **IAM Permissions**:
Ensure your IAM role or user has the following permissions:

- `bedrock:InvokeModel`
- `bedrock:ListFoundationModels`
- `bedrock:GetFoundationModel`

## Usage

### Running the Application

```bash
go run main.go
```

By default, the application will:

1. Connect to AWS Bedrock using your credentials
2. Send the configured prompt to the Llama 3 model
3. Display the model's response
4. Show token usage information if available

### Configuration Options

You can customize the behavior by modifying the following constants in `main.go`:

```go
const (
    // AWS region to use
    awsRegion = "us-east-2"

    // Model ID or inference profile ARN
    modelID = "arn:aws:bedrock:us-east-2:913524932967:inference-profile/us.meta.llama3-2-1b-instruct-v1:0"

    // Prompt to send to the model
    prompt = "Hello, how are you"
)
```

### Model Parameters

The following parameters can be adjusted in the `Payload` struct:

- `MaxGenLen`: Maximum length of the generated response (default: 512)
- `Temperature`: Controls randomness in the output (default: 0.7)
- `TopP`: Controls diversity via nucleus sampling (default: 0.9)

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
