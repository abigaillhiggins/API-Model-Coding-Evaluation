"""
Code Generation and Problem Solving Evaluation Suite
Tests an LLM's ability to write, debug, and optimize code across different scenarios.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
import json
from abc import ABC, abstractmethod
import logging
import ast
import time
import subprocess
from pathlib import Path
import tempfile
import os
import re
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure rate limiting
RATE_LIMIT_CALLS = int(os.getenv("RATE_LIMIT_CALLS", "60"))
RATE_LIMIT_PERIOD = int(os.getenv("RATE_LIMIT_PERIOD", "60"))

class RateLimiter:
    """Simple rate limiter implementation"""
    def __init__(self, calls: int, period: float):
        self.calls = calls
        self.period = period
        self.timestamps = []

    async def acquire(self):
        now = time.time()
        # Remove old timestamps
        self.timestamps = [ts for ts in self.timestamps if now - ts < self.period]
        
        if len(self.timestamps) >= self.calls:
            # Wait until the oldest timestamp is outside the period
            wait_time = self.timestamps[0] + self.period - now
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Recursive call after waiting
                await self.acquire()
        
        self.timestamps.append(now)

    async def __aenter__(self):
        await self.acquire()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

rate_limiter = RateLimiter(RATE_LIMIT_CALLS, RATE_LIMIT_PERIOD)

def extract_code_from_response(response: str) -> str:
    """
    Extracts clean Python code from a response that might contain markdown or explanations.
    Handles both markdown code blocks and raw code.
    """
    # Try to find Python code blocks
    code_block_pattern = r"```(?:python)?\s*(.*?)\s*```"
    matches = re.findall(code_block_pattern, response, re.DOTALL)
    
    if matches:
        # Return the first code block found
        return matches[0].strip()
    
    # If no code blocks found, try to extract code by looking for function definitions
    code_lines = []
    in_code = False
    
    for line in response.split('\n'):
        if line.strip().startswith('def ') or line.strip().startswith('class '):
            in_code = True
        
        if in_code:
            code_lines.append(line)
    
    if code_lines:
        return '\n'.join(code_lines)
    
    # If no clear code sections found, return the original response
    # stripped of any obvious documentation
    return '\n'.join(line for line in response.split('\n') 
                    if not line.strip().startswith(('#', '"""', "'''", '//', '-', '>')))

@dataclass
class CodeChallenge:
    """Represents a coding challenge"""
    name: str
    description: str
    difficulty: str  # 'easy', 'medium', 'hard'
    category: str  # 'algorithms', 'debugging', 'optimization', etc.
    prompt: str
    test_cases: List[Dict[str, Any]]
    expected_elements: List[str]
    time_limit: float  # seconds
    memory_limit: int  # MB

@dataclass
class CodeResponse:
    """Represents a response to a coding challenge"""
    code: str
    language: str
    explanation: str
    complexity_analysis: Optional[str] = None
    runtime: Optional[float] = None
    memory_usage: Optional[int] = None

class BaseProvider(ABC):
    """Base class for LLM providers"""
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        
    async def __aenter__(self):
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass
        
    @abstractmethod
    async def generate_code(self, prompt: str, **kwargs) -> CodeResponse:
        """Generate code from prompt"""
        pass
        
    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass

class CodeEvaluator:
    """Evaluates code responses"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def _analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyze code quality metrics"""
        try:
            # Parse the code
            tree = ast.parse(code)
            
            # Initialize metrics
            metrics = {
                "complexity": 0,
                "num_functions": 0,
                "num_classes": 0,
                "num_lines": len(code.splitlines()),
                "has_docstrings": False,
                "has_type_hints": False,
                "expected_elements_found": 0
            }
            
            # Analyze AST
            for node in ast.walk(tree):
                # Count functions
                if isinstance(node, ast.FunctionDef):
                    metrics["num_functions"] += 1
                    if ast.get_docstring(node):
                        metrics["has_docstrings"] = True
                    if any(isinstance(n, ast.AnnAssign) for n in ast.walk(node)):
                        metrics["has_type_hints"] = True
                
                # Count classes
                elif isinstance(node, ast.ClassDef):
                    metrics["num_classes"] += 1
                
                # Count loops and conditionals for complexity
                elif isinstance(node, (ast.For, ast.While, ast.If, ast.Try)):
                    metrics["complexity"] += 1
            
            return metrics
            
        except Exception as e:
            logger.warning(f"Error analyzing code quality: {str(e)}")
            return {}
    
    async def evaluate_python_code(self, challenge: CodeChallenge, response: CodeResponse) -> Dict[str, Any]:
        """Evaluates Python code response"""
        results = {
            "passed_tests": 0,
            "total_tests": len(challenge.test_cases),
            "runtime_performance": [],
            "memory_usage": [],
            "code_quality": {},
            "errors": []
        }
        
        # Create temporary Python file
        test_file = self.temp_dir / "test_solution.py"
        
        try:
            # Write code to file
            with open(test_file, "w") as f:
                f.write(response.code)
            
            # Log the generated code
            logger.info(f"Generated code:\n{response.code}")
            
            # Static Analysis
            results["code_quality"] = self._analyze_code_quality(response.code)
            
            # Run test cases
            for test_case in challenge.test_cases:
                try:
                    start_time = time.time()
                    # Execute in subprocess for safety and memory tracking
                    process = subprocess.run(
                        ["python", str(test_file)],
                        input=json.dumps(test_case["input"]),
                        text=True,
                        capture_output=True,
                        timeout=challenge.time_limit
                    )
                    
                    runtime = time.time() - start_time
                    results["runtime_performance"].append(runtime)
                    
                    # Log test case results
                    logger.info(f"Test case input: {test_case['input']}")
                    logger.info(f"Expected output: {test_case['expected']}")
                    logger.info(f"Actual output: {process.stdout.strip()}")
                    
                    # Compare output
                    if process.stdout.strip() == str(test_case["expected"]).strip():
                        results["passed_tests"] += 1
                    
                except subprocess.TimeoutExpired:
                    results["errors"].append(f"Time limit exceeded for test case: {test_case['input']}")
                except Exception as e:
                    results["errors"].append(f"Error running test case: {str(e)}")
            
        except Exception as e:
            results["errors"].append(f"Evaluation error: {str(e)}")
        
        return results

class CodeEvaluationSuite:
    """Main evaluation suite that coordinates providers and evaluations"""
    
    def __init__(self, provider: BaseProvider):
        self.provider = provider
        self.evaluator = CodeEvaluator()
    
    def _calculate_score(self, evaluation: Dict[str, Any], challenge: CodeChallenge) -> float:
        """Calculate a score between 0 and 1 for the evaluation results"""
        score = 0.0
        
        # Test cases passed (50% of score)
        if evaluation["total_tests"] > 0:
            test_score = evaluation["passed_tests"] / evaluation["total_tests"]
            score += 0.5 * test_score
        
        # Code quality (30% of score)
        if evaluation["code_quality"]:
            quality_score = 0.0
            quality_metrics = evaluation["code_quality"]
            
            # Check for expected elements (e.g., functions, loops)
            if "expected_elements_found" in quality_metrics:
                quality_score += 0.15 * (quality_metrics["expected_elements_found"] / len(challenge.expected_elements))
            
            # No errors is good
            if not evaluation["errors"]:
                quality_score += 0.15
                
            score += quality_score
        
        # Performance (20% of score)
        if evaluation["runtime_performance"]:
            avg_runtime = sum(evaluation["runtime_performance"]) / len(evaluation["runtime_performance"])
            if avg_runtime <= challenge.time_limit:
                # Full performance score if within time limit
                score += 0.2
            else:
                # Partial score based on how close to time limit
                score += 0.2 * (challenge.time_limit / avg_runtime)
        
        return min(1.0, max(0.0, score))
    
    async def run_evaluation(self, challenges: Optional[List[CodeChallenge]] = None) -> Dict[str, Any]:
        """Run the evaluation suite for a single provider"""
        if challenges is None:
            challenges = create_coding_challenges()
            
        results = {
            "provider_info": await self.provider.get_model_info(),
            "challenges": [],
            "summary": {
                "total_challenges": len(challenges),
                "passed_challenges": 0,
                "average_runtime": 0.0,
                "average_memory": 0.0
            }
        }
        
        total_runtime = 0.0
        total_memory = 0.0
        
        async with self.provider:  # Use context manager for provider
            for challenge in challenges:
                try:
                    async with rate_limiter:
                        response = await self.provider.generate_code(challenge.prompt)
                    evaluation = await self.evaluator.evaluate_python_code(challenge, response)
                    
                    challenge_result = {
                        "name": challenge.name,
                        "difficulty": challenge.difficulty,
                        "category": challenge.category,
                        "evaluation": evaluation,
                        "score": self._calculate_score(evaluation, challenge)
                    }
                    
                    if evaluation["passed_tests"] == len(challenge.test_cases):
                        results["summary"]["passed_challenges"] += 1
                    
                    if evaluation["runtime_performance"]:
                        avg_runtime = sum(evaluation["runtime_performance"]) / len(evaluation["runtime_performance"])
                        total_runtime += avg_runtime
                        
                    if evaluation["memory_usage"]:
                        avg_memory = sum(evaluation["memory_usage"]) / len(evaluation["memory_usage"])
                        total_memory += avg_memory
                    
                    results["challenges"].append(challenge_result)
                    
                except Exception as e:
                    logger.error(f"Error evaluating challenge {challenge.name}: {str(e)}")
                    results["challenges"].append({
                        "name": challenge.name,
                        "error": str(e)
                    })
        
        # Calculate averages
        num_challenges = len(challenges)
        if num_challenges > 0:
            results["summary"]["average_runtime"] = total_runtime / num_challenges
            results["summary"]["average_memory"] = total_memory / num_challenges
        
        return results

    @staticmethod
    async def run_multi_provider_evaluation(providers: List[BaseProvider], challenges: Optional[List[CodeChallenge]] = None) -> Dict[str, Any]:
        """Run evaluations across multiple providers and compare results"""
        if challenges is None:
            challenges = create_coding_challenges()

        results = {
            "summary": {
                "total_providers": len(providers),
                "total_challenges": len(challenges),
                "best_provider": None,
                "average_scores": {}
            },
            "provider_results": {}
        }

        # Run evaluations for each provider
        for provider in providers:
            suite = CodeEvaluationSuite(provider)
            provider_result = await suite.run_evaluation(challenges)
            
            # Calculate average score for this provider
            total_score = 0.0
            challenge_count = 0
            for challenge in provider_result["challenges"]:
                if "score" in challenge:
                    total_score += challenge["score"]
                    challenge_count += 1
            
            avg_score = total_score / challenge_count if challenge_count > 0 else 0.0
            results["summary"]["average_scores"][provider.model_name] = avg_score
            
            # Store full results
            results["provider_results"][provider.model_name] = provider_result

        # Determine best provider
        if results["summary"]["average_scores"]:
            best_provider = max(results["summary"]["average_scores"].items(), key=lambda x: x[1])
            results["summary"]["best_provider"] = {
                "model": best_provider[0],
                "average_score": best_provider[1]
            }

        return results

def create_coding_challenges() -> List[CodeChallenge]:
    """Create a list of coding challenges"""
    return [
        CodeChallenge(
            name="Binary Search Implementation",
            description="Implement a binary search algorithm",
            difficulty="easy",
            category="algorithms",
            prompt="Implement a binary search function that takes a sorted list and a target value as input and returns the index of the target value if found, or -1 if not found.",
            test_cases=[
                {"input": {"arr": [1, 3, 5, 7, 9], "target": 5}, "expected": 2},
                {"input": {"arr": [1, 3, 5, 7, 9], "target": 6}, "expected": -1},
            ],
            expected_elements=["def binary_search", "while", "return"],
            time_limit=1.0,
            memory_limit=128
        ),
        # Add more challenges here
    ]

async def main():
    """Main entry point"""
    from llm_providers import get_provider
    
    # Example of running single provider (default behavior)
    provider_name = os.getenv("DEFAULT_PROVIDER", "openai")
    model_name = os.getenv("DEFAULT_MODEL", "gpt-4-0125-preview")
    
    if os.getenv("TEST_ALL_PROVIDERS", "false").lower() == "true":
        # Test all providers that have API keys configured
        providers = []
        provider_configs = {
            "deepseek": ("DEEPSEEK_API_KEY", "deepseek-ai/deepseek-coder-33b-instruct"),
            "openai": ("OPENAI_API_KEY", "gpt-4-0125-preview"),
            "together": ("TOGETHER_API_KEY", "codellama/CodeLlama-34b-Python-hf"),
            "anthropic": ("ANTHROPIC_API_KEY", "claude-3-opus-20240229"),
            "gemini": ("GOOGLE_API_KEY", "gemini-pro")
        }
        
        for provider_name, (key_env, default_model) in provider_configs.items():
            if os.getenv(key_env):
                try:
                    provider = get_provider(provider_name, model_name=default_model)
                    providers.append(provider)
                    logger.info(f"Successfully initialized {provider_name} with model {default_model}")
                except Exception as e:
                    logger.warning(f"Could not initialize {provider_name}: {str(e)}")
        
        if not providers:
            logger.error("No providers could be initialized. Please check your API keys in .env")
            return
        
        logger.info(f"Running evaluation with {len(providers)} providers: {[p.model_name for p in providers]}")
        results = await CodeEvaluationSuite.run_multi_provider_evaluation(providers)
    else:
        # Run single provider
        provider = get_provider(provider_name, model_name=model_name)
        suite = CodeEvaluationSuite(provider)
        results = await suite.run_evaluation()
    
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 
