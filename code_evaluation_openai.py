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

class CodeEvaluator:
    """Evaluates code responses"""
    
    def __init__(self):
        self.temp_dir = Path(tempfile.mkdtemp())
    
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
    
    def _analyze_code_quality(self, code: str) -> Dict[str, Any]:
        """Analyzes code quality metrics"""
        try:
            tree = ast.parse(code)
            
            return {
                "complexity": self._calculate_complexity(tree),
                "line_count": len(code.splitlines()),
                "function_count": len([node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]),
                "class_count": len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]),
                "comment_ratio": self._calculate_comment_ratio(code),
                "variable_naming": self._analyze_variable_naming(tree)
            }
        except Exception as e:
            logger.error(f"Error in code quality analysis: {str(e)}")
            return {}
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculates cyclomatic complexity"""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    def _calculate_comment_ratio(self, code: str) -> float:
        """Calculates ratio of comments to code"""
        lines = code.splitlines()
        comment_lines = sum(1 for line in lines if line.strip().startswith('#'))
        return comment_lines / len(lines) if lines else 0
    
    def _analyze_variable_naming(self, tree: ast.AST) -> Dict[str, Any]:
        """Analyzes variable naming conventions"""
        variables = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                variables.append(node.id)
        
        return {
            "snake_case_ratio": sum(1 for var in variables if '_' in var) / len(variables) if variables else 0,
            "avg_length": sum(len(var) for var in variables) / len(variables) if variables else 0
        }

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers"""
    async def generate_code(self, prompt: str, **kwargs) -> CodeResponse:
        """Generate code from prompt"""
        ...

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        ...

class BaseProvider(ABC):
    """Base class for LLM providers"""
    def __init__(self, api_key: str, model_name: str):
        self.api_key = api_key
        self.model_name = model_name
        self._session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

    @abstractmethod
    async def generate_code(self, prompt: str, **kwargs) -> CodeResponse:
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        pass

class DeepSeekProvider(BaseProvider):
    """DeepSeek implementation"""
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deepseek-coder"):
        super().__init__(api_key or os.getenv("DEEPSEEK_API_KEY"), model_name)
        if not self.api_key:
            raise ValueError("DeepSeek API key must be provided either through constructor or DEEPSEEK_API_KEY environment variable")
        self.base_url = "https://api.deepseek.com/v1"

    async def generate_code(self, prompt: str, **kwargs) -> CodeResponse:
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        # Add specific instructions about code format
        formatted_prompt = f"""
        IMPORTANT: Provide ONLY the Python code implementation. Do not include markdown formatting, documentation blocks, or explanations.
        The code should be directly executable.

        {prompt}
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": formatted_prompt}],
            **kwargs
        }
        
        async with self._session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Clean the response and extract pure code
            clean_code = extract_code_from_response(content)
            
            return CodeResponse(
                code=clean_code,
                language="python",
                explanation=""  # We're not using explanations in this version
            )

    async def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.model_name,
            "type": "code"
        }

class OpenAIProvider(BaseProvider):
    """OpenAI implementation"""
    AVAILABLE_MODELS = [
        "gpt-4-turbo-preview",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k"
    ]
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-turbo-preview"):
        if model_name not in self.AVAILABLE_MODELS:
            raise ValueError(f"Model {model_name} not in available models: {self.AVAILABLE_MODELS}")
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"), model_name)
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided either through constructor or OPENAI_API_KEY environment variable")
        self.base_url = "https://api.openai.com/v1"

    async def generate_code(self, prompt: str, **kwargs) -> CodeResponse:
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        # Add specific instructions about code format
        formatted_prompt = f"""
        IMPORTANT: Provide ONLY the Python code implementation. Do not include markdown formatting, documentation blocks, or explanations.
        The code should be directly executable.

        {prompt}
        """
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": "You are a Python coding assistant. Provide only executable Python code without explanations or markdown."},
                {"role": "user", "content": formatted_prompt}
            ],
            "temperature": 0.7,
            **kwargs
        }
        
        async with self._session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Clean the response and extract pure code
            clean_code = extract_code_from_response(content)
            
            return CodeResponse(
                code=clean_code,
                language="python",
                explanation=""  # We're not using explanations in this version
            )

    async def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model_name,
            "type": "code"
        }

def create_coding_challenges() -> List[CodeChallenge]:
    """Creates a list of coding challenges"""
    return [
        CodeChallenge(
            name="Binary Search Implementation",
            description="Implement an efficient binary search algorithm",
            difficulty="medium",
            category="algorithms",
            prompt="""
            Implement a binary search function that finds the index of a target value in a sorted array.
            If the target is not found, return -1.
            
            Example:
            Input: arr = [1, 3, 5, 7, 9], target = 5
            Output: 2
            
            Requirements:
            - Handle edge cases
            - Include time/space complexity analysis
            - Add clear comments
            - Use proper variable naming
            """,
            test_cases=[
                {"input": {"arr": [1, 3, 5, 7, 9], "target": 5}, "expected": 2},
                {"input": {"arr": [1, 3, 5, 7, 9], "target": 1}, "expected": 0},
                {"input": {"arr": [1, 3, 5, 7, 9], "target": 10}, "expected": -1},
                {"input": {"arr": [], "target": 5}, "expected": -1}
            ],
            expected_elements=[
                "binary search logic",
                "edge case handling",
                "comments",
                "complexity analysis"
            ],
            time_limit=1.0,
            memory_limit=128
        ),
        CodeChallenge(
            name="String Manipulation",
            description="Implement a function to find the longest palindromic substring",
            difficulty="hard",
            category="algorithms",
            prompt="""
            Implement a function that finds the longest palindromic substring in a given string.
            
            Example:
            Input: "babad"
            Output: "bab" or "aba"
            
            Requirements:
            - Optimize for efficiency
            - Handle edge cases
            - Include complexity analysis
            - Add comprehensive comments
            """,
            test_cases=[
                {"input": {"s": "babad"}, "expected": "bab"},
                {"input": {"s": "cbbd"}, "expected": "bb"},
                {"input": {"s": "a"}, "expected": "a"},
                {"input": {"s": ""}, "expected": ""}
            ],
            expected_elements=[
                "palindrome check",
                "optimization",
                "edge cases",
                "comments"
            ],
            time_limit=2.0,
            memory_limit=256
        )
    ]

class CodeEvaluationSuite:
    """Main evaluation suite for testing LLM coding capabilities"""
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.evaluator = CodeEvaluator()
        self.challenges = create_coding_challenges()
        self.results: Dict[str, Any] = {}

    async def run_evaluation(self) -> Dict[str, Any]:
        """Runs the complete evaluation suite"""
        logger.info("Starting code evaluation...")
        
        challenge_results = []
        total_score = 0.0
        completed_challenges = 0
        
        for challenge in self.challenges:
            try:
                logger.info(f"Evaluating challenge: {challenge.name}")
                
                # Generate code
                response = await self.provider.generate_code(challenge.prompt)
                logger.info("Code generated successfully")
                
                # Evaluate response
                evaluation = await self.evaluator.evaluate_python_code(challenge, response)
                logger.info("Code evaluation completed")
                
                try:
                    # Calculate score
                    score = self._calculate_score(evaluation, challenge)
                    logger.info(f"Score calculated: {score}")
                    total_score += score
                    completed_challenges += 1
                    
                    challenge_results.append({
                        "challenge_name": challenge.name,
                        "difficulty": challenge.difficulty,
                        "category": challenge.category,
                        "score": score,
                        "evaluation": {
                            "passed_tests": evaluation["passed_tests"],
                            "total_tests": evaluation["total_tests"],
                            "runtime_performance": evaluation["runtime_performance"],
                            "code_quality": {
                                "complexity": evaluation["code_quality"].get("complexity", 0),
                                "comment_ratio": evaluation["code_quality"].get("comment_ratio", 0),
                                "variable_naming": evaluation["code_quality"].get("variable_naming", {})
                            } if evaluation["code_quality"] else {}
                        },
                        "response": {
                            "code": response.code
                        }
                    })
                    
                except Exception as calc_error:
                    logger.error(f"Error calculating score: {str(calc_error)}")
                    challenge_results.append({
                        "challenge_name": challenge.name,
                        "error": f"Score calculation error: {str(calc_error)}"
                    })
                
            except Exception as e:
                logger.error(f"Error evaluating challenge {challenge.name}: {str(e)}")
                challenge_results.append({
                    "challenge_name": challenge.name,
                    "error": str(e)
                })
        
        avg_score = total_score / completed_challenges if completed_challenges > 0 else 0.0
        
        self.results = {
            "model_info": await self.provider.get_model_info(),
            "total_score": avg_score,
            "challenge_results": challenge_results,
            "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "completed_challenges": completed_challenges,
            "total_challenges": len(self.challenges)
        }
        
        return self.results

    def _calculate_score(self, evaluation: Dict[str, Any], challenge: CodeChallenge) -> float:
        """Calculates score for a challenge response"""
        score = 0.0
        
        try:
            # Test cases (40%)
            test_score = (evaluation["passed_tests"] / evaluation["total_tests"]) * 40
            score += test_score
            logger.debug(f"Test score: {test_score}")
            
            # Code quality (30%)
            quality_score = 0
            if evaluation["code_quality"]:
                # Complexity score (10 points)
                complexity = evaluation["code_quality"].get("complexity", 0)
                if complexity < 5:
                    quality_score += 10
                elif complexity < 10:
                    quality_score += 5
                
                # Comment ratio score (10 points)
                comment_ratio = evaluation["code_quality"].get("comment_ratio", 0)
                if comment_ratio > 0.1:
                    quality_score += 10
                
                # Variable naming score (10 points)
                variable_naming = evaluation["code_quality"].get("variable_naming", {})
                snake_case_ratio = variable_naming.get("snake_case_ratio", 0)
                if snake_case_ratio > 0.8:
                    quality_score += 10
            
            score += quality_score
            logger.debug(f"Quality score: {quality_score}")
            
            # Performance (30%)
            performance_score = 0
            if evaluation["runtime_performance"]:
                avg_runtime = sum(evaluation["runtime_performance"]) / len(evaluation["runtime_performance"])
                if avg_runtime < challenge.time_limit / 2:
                    performance_score = 30
                elif avg_runtime < challenge.time_limit:
                    performance_score = 15
            
            score += performance_score
            logger.debug(f"Performance score: {performance_score}")
            
        except Exception as e:
            logger.error(f"Error in score calculation: {str(e)}")
            raise
        
        return min(score, 100.0)

async def evaluate_all_openai_models():
    """Evaluates all available OpenAI models"""
    results = {}
    
    for model in OpenAIProvider.AVAILABLE_MODELS:
        print(f"\nEvaluating model: {model}")
        print("=" * 50)
        
        try:
            async with OpenAIProvider(model_name=model) as provider:
                suite = CodeEvaluationSuite(provider)
                model_results = await suite.run_evaluation()
                results[model] = model_results
                
                # Print individual model results
                print(f"\nResults for {model}:")
                print("-" * 50)
                print(f"Overall Score: {model_results['total_score']:.2f}")
                print(f"Completed: {model_results['completed_challenges']}/{model_results['total_challenges']} challenges")
                
                for challenge in model_results["challenge_results"]:
                    print(f"\n{challenge['challenge_name']} ({challenge.get('difficulty', 'unknown')}):")
                    if "error" in challenge:
                        print(f"Error: {challenge['error']}")
                    else:
                        print(f"Score: {challenge['score']:.2f}")
                        if "evaluation" in challenge:
                            eval_data = challenge["evaluation"]
                            print(f"Passed Tests: {eval_data['passed_tests']}/{eval_data['total_tests']}")
        
        except Exception as e:
            print(f"Error evaluating {model}: {str(e)}")
            results[model] = {"error": str(e)}
    
    # Print comparative results
    print("\nComparative Results")
    print("=" * 50)
    print(f"{'Model':<20} {'Score':<10} {'Completed':<15}")
    print("-" * 50)
    
    for model, result in results.items():
        if isinstance(result.get("total_score"), (int, float)):
            score_str = f"{result['total_score']:.2f}"
        else:
            score_str = "N/A"
        completed = f"{result.get('completed_challenges', 0)}/{result.get('total_challenges', 0)}"
        print(f"{model:<20} {score_str:<10} {completed:<15}")
    
    return results

async def main():
    """Main function to demonstrate usage"""
    # Try to get API key from environment variable
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set it using:")
        print("export OPENAI_API_KEY='your-api-key'")
        return
    
    try:
        await evaluate_all_openai_models()
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        logger.exception("Detailed error information:")

if __name__ == "__main__":
    asyncio.run(main()) 