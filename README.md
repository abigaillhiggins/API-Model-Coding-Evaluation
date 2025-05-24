# Code Generation and Evaluation Suite

This project implements a comprehensive evaluation system for testing various code-specialized Large Language Models (LLMs) on coding tasks. It supports multiple providers including OpenAI GPT-4, Anthropic Claude 3, DeepSeek Coder, Together.ai's Code Llama, and Google's Gemini.

## Features

- Multi-provider support for code generation evaluation
- Automated test case execution
- Code quality analysis using AST parsing
- Performance metrics collection
- Rate limiting and API error handling
- Scoring system based on:
  - Test case success (50%)
  - Code quality (30%)
  - Performance (20%)

## Supported Providers

- OpenAI (GPT-4)
- Anthropic (Claude 3)
- DeepSeek Coder
- Together.ai (Code Llama)
- Google (Gemini)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/abigaillhiggins/CA_test_repo.git
cd CA_test_repo
```

2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Create a `.env` file with your API keys:
```
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
TOGETHER_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
```

## Usage

### Single Provider Testing
```bash
python code_evaluation.py
```

### Multi-Provider Testing
```bash
export TEST_ALL_PROVIDERS=true
python code_evaluation.py
```

## Configuration

- Default provider and model can be configured in the `.env` file
- Rate limiting can be adjusted via environment variables:
  - `RATE_LIMIT_CALLS`: Number of calls allowed per period
  - `RATE_LIMIT_PERIOD`: Time period in seconds

## License

MIT License 