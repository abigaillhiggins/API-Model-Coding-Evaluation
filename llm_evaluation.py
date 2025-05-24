"""
Language Model Evaluation Suite
A framework for evaluating any LLM's capabilities across different dimensions
of software development, focusing on news aggregation and content transformation.
"""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional, Protocol, runtime_checkable
from dataclasses import dataclass
from datetime import datetime
import json
from abc import ABC, abstractmethod
import logging
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    COHERE = "cohere"
    DEEPSEEK = "deepseek"
    CUSTOM = "custom"

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers"""
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        ...

    async def embed(self, text: str) -> List[float]:
        """Generate embeddings for text"""
        ...

    @property
    def model_info(self) -> Dict[str, Any]:
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
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def embed(self, text: str) -> List[float]:
        pass

    @property
    @abstractmethod
    def model_info(self) -> Dict[str, Any]:
        pass

class OpenAIProvider(BaseProvider):
    """OpenAI API provider implementation"""
    async def generate(self, prompt: str, **kwargs) -> str:
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        
        async with self._session.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["message"]["content"]

    async def embed(self, text: str) -> List[float]:
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "text-embedding-ada-002",
            "input": text
        }
        
        async with self._session.post(
            "https://api.openai.com/v1/embeddings",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return result["data"][0]["embedding"]

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "provider": "openai",
            "model": self.model_name,
            "type": "chat"
        }

class DeepSeekProvider(BaseProvider):
    """DeepSeek API provider implementation"""
    def __init__(self, api_key: str, model_name: str = "deepseek-chat"):
        super().__init__(api_key, model_name)
        self.base_url = "https://api.deepseek.com/v1"

    async def generate(self, prompt: str, **kwargs) -> str:
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs
        }
        
        async with self._session.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return result["choices"][0]["message"]["content"]

    async def embed(self, text: str) -> List[float]:
        if not self._session:
            raise RuntimeError("Session not initialized. Use 'async with' context manager.")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        data = {
            "model": "deepseek-embed",
            "input": text
        }
        
        async with self._session.post(
            f"{self.base_url}/embeddings",
            headers=headers,
            json=data
        ) as response:
            response.raise_for_status()
            result = await response.json()
            return result["data"][0]["embedding"]

    @property
    def model_info(self) -> Dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.model_name,
            "type": "chat"
        }

@dataclass
class EvaluationTask:
    """Represents a single evaluation task"""
    name: str
    description: str
    prompt: str
    expected_elements: List[str]
    scoring_criteria: Dict[str, float]
    
    def score_response(self, response: str) -> float:
        """Score the model's response based on criteria"""
        score = 0.0
        for element, points in self.scoring_criteria.items():
            if element.lower() in response.lower():
                score += points
        return min(score, 100.0)

class EvaluationSuite:
    """Main evaluation suite for testing LLM capabilities"""
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.tasks: List[EvaluationTask] = []
        self.results: Dict[str, Any] = {}

    def add_task(self, task: EvaluationTask):
        """Add an evaluation task"""
        self.tasks.append(task)

    async def run_evaluation(self) -> Dict[str, Any]:
        """Run all evaluation tasks"""
        total_score = 0
        task_results = {}

        for task in self.tasks:
            try:
                logger.info(f"Running task: {task.name}")
                response = await self.provider.generate(task.prompt)
                score = task.score_response(response)
                
                task_results[task.name] = {
                    "score": score,
                    "response": response
                }
                total_score += score
                
            except Exception as e:
                logger.error(f"Error in task {task.name}: {str(e)}")
                task_results[task.name] = {
                    "score": 0,
                    "error": str(e)
                }

        self.results = {
            "model_info": self.provider.model_info,
            "total_score": total_score / len(self.tasks),
            "task_results": task_results
        }
        
        return self.results

def create_default_tasks() -> List[EvaluationTask]:
    """Create default evaluation tasks"""
    return [
        EvaluationTask(
            name="news_style_transformation",
            description="Transform news article into Gossip Girl style",
            prompt="""
            Transform this news article into Gossip Girl style narration:
            
            Title: Apple Announces New iPhone
            Content: Apple Inc. unveiled its latest iPhone model today, featuring a titanium frame
            and improved camera system. The device will be available next month starting at $999.
            """,
            expected_elements=[
                "XOXO",
                "Upper East Side",
                "Spotted",
                "fashion reference",
                "dramatic tone"
            ],
            scoring_criteria={
                "XOXO": 20,
                "Spotted": 20,
                "fashion reference": 20,
                "dramatic tone": 20,
                "social status reference": 20
            }
        ),
        # Add more tasks here
    ]

async def main():
    """Main function to demonstrate usage"""
    # Example with OpenAI
    api_key = "your-api-key"  # Replace with actual API key
    
    async with OpenAIProvider(api_key, "gpt-4") as provider:
        suite = EvaluationSuite(provider)
        
        # Add default tasks
        for task in create_default_tasks():
            suite.add_task(task)
        
        # Run evaluation
        results = await suite.run_evaluation()
        
        # Print results
        print("\nEvaluation Results:")
        print(json.dumps(results, indent=2))

if __name__ == "__main__":
    asyncio.run(main()) 