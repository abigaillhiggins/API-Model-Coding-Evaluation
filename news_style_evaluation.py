"""
News Processing and Style Transformation Evaluation Suite
A framework for evaluating any LLM's capabilities in news processing and creative rewriting.
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
import feedparser
import html2text

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class NewsArticle:
    """Represents a news article"""
    title: str
    summary: str
    url: str
    published_date: datetime
    source: str = "NYT"

@runtime_checkable
class LLMProvider(Protocol):
    """Protocol defining the interface for LLM providers"""
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from prompt"""
        ...

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        ...

class NewsProcessor:
    """Handles news fetching and processing"""
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.h2t = html2text.HTML2Text()
        self.h2t.ignore_links = True
        
    async def fetch_latest_news(self, count: int = 3) -> List[NewsArticle]:
        """Fetch latest news from NYT RSS feed"""
        # Using RSS feed as it doesn't require API key
        nyt_feed = "https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml"
        
        try:
            feed = feedparser.parse(nyt_feed)
            articles = []
            
            for entry in feed.entries[:count]:
                article = NewsArticle(
                    title=self.h2t.handle(entry.title),
                    summary=self.h2t.handle(entry.summary),
                    url=entry.link,
                    published_date=datetime.strptime(entry.published, "%a, %d %b %Y %H:%M:%S %z"),
                )
                articles.append(article)
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []

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
    async def get_model_info(self) -> Dict[str, Any]:
        pass

@dataclass
class StyleTransformationTask:
    """Represents a style transformation task"""
    name: str
    description: str
    source_text: str
    target_style: str
    style_elements: List[str]
    scoring_criteria: Dict[str, float]

    def score_response(self, response: str) -> float:
        """Score the model's response based on criteria"""
        score = 0.0
        for element, points in self.scoring_criteria.items():
            if element.lower() in response.lower():
                score += points
        return min(score, 100.0)

class NewsStyleEvaluator:
    """Main evaluation suite for testing LLM capabilities in news processing"""
    def __init__(self, provider: LLMProvider):
        self.provider = provider
        self.news_processor = NewsProcessor()
        self.results: Dict[str, Any] = {}

    def create_gossip_girl_prompt(self, article: NewsArticle) -> str:
        """Create a prompt for Gossip Girl style transformation"""
        return f"""
        Transform this news article into Gossip Girl style narration with witty, sassy language:

        Title: {article.title}
        Content: {article.summary}
        
        Requirements:
        - Use the signature Gossip Girl voice and tone
        - Include "XOXO, Gossip Girl" signature
        - Reference Upper East Side culture and fashion
        - Add dramatic flair and social commentary
        - Keep the core news information intact
        """

    async def evaluate_article_transformation(self, article: NewsArticle) -> Dict[str, Any]:
        """Evaluate LLM's ability to transform a single article"""
        prompt = self.create_gossip_girl_prompt(article)
        
        try:
            response = await self.provider.generate(prompt)
            
            task = StyleTransformationTask(
                name="gossip_girl_transformation",
                description="Transform news into Gossip Girl style",
                source_text=article.summary,
                target_style="Gossip Girl",
                style_elements=[
                    "XOXO",
                    "Upper East Side",
                    "Spotted",
                    "fashion reference",
                    "dramatic tone",
                    "social commentary",
                    "signature phrases",
                    "sassy language"
                ],
                scoring_criteria={
                    "XOXO": 10,
                    "Spotted": 10,
                    "Upper East Side reference": 10,
                    "fashion reference": 10,
                    "dramatic tone": 15,
                    "social commentary": 15,
                    "signature phrases": 15,
                    "sassy language": 15
                }
            )
            
            score = task.score_response(response)
            
            return {
                "original_title": article.title,
                "transformed_content": response,
                "score": score,
                "url": article.url,
                "published_date": article.published_date.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in transformation: {str(e)}")
            return {
                "original_title": article.title,
                "error": str(e),
                "score": 0
            }

    async def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation suite"""
        logger.info("Starting news transformation evaluation...")
        
        # Fetch latest news
        articles = await self.news_processor.fetch_latest_news(3)
        if not articles:
            raise ValueError("Failed to fetch news articles")
        
        # Transform each article
        transformations = []
        total_score = 0
        
        for article in articles:
            result = await self.evaluate_article_transformation(article)
            transformations.append(result)
            total_score += result.get("score", 0)
        
        # Compile results
        self.results = {
            "model_info": await self.provider.get_model_info(),
            "total_score": total_score / len(articles),
            "transformations": transformations,
            "evaluation_timestamp": datetime.now().isoformat()
        }
        
        return self.results

# Example provider implementation for DeepSeek
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

    async def get_model_info(self) -> Dict[str, Any]:
        return {
            "provider": "deepseek",
            "model": self.model_name,
            "type": "chat"
        }

async def main():
    """Main function to demonstrate usage"""
    api_key = "your-api-key"  # Replace with actual API key
    
    try:
        async with DeepSeekProvider(api_key) as provider:
            evaluator = NewsStyleEvaluator(provider)
            results = await evaluator.run_evaluation()
            
            # Print results
            print("\nEvaluation Results:")
            print("-" * 50)
            print(f"Model: {results['model_info']['model']}")
            print(f"Provider: {results['model_info']['provider']}")
            print(f"Overall Score: {results['total_score']:.2f}")
            print("\nTransformations:")
            
            for i, transform in enumerate(results['transformations'], 1):
                print(f"\nArticle {i}:")
                print(f"Original Title: {transform['original_title']}")
                print(f"Score: {transform['score']:.2f}")
                print("Transformed Content:")
                print("-" * 30)
                print(transform['transformed_content'])
                print("-" * 30)
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 