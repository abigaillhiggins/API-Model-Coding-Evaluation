"""
LLM Provider implementations for various code-specialized models.
"""

import os
from typing import Dict, Any, Optional, List
import aiohttp
import json
from abc import ABC
import anthropic
import openai
import together
from tenacity import retry, stop_after_attempt, wait_exponential
import google.generativeai as genai

from code_evaluation import BaseProvider, CodeResponse

class DeepSeekProvider(BaseProvider):
    """Provider for DeepSeek models"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "deepseek-ai/deepseek-coder-33b-instruct"):
        super().__init__(api_key or os.getenv("DEEPSEEK_API_KEY"), model_name)
        self.base_url = "https://api.deepseek.com/v1"
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers={"Authorization": f"Bearer {self.api_key}"})
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            self.session = None

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_code(self, prompt: str) -> CodeResponse:
        """Generate code using the DeepSeek API"""
        if not self.session:
            raise RuntimeError("Session not initialized. Use async with context manager.")
            
        data = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        async with self.session.post(f"{self.base_url}/chat/completions", json=data) as response:
            if response.status != 200:
                error_text = await response.text()
                raise Exception(f"DeepSeek API error: {error_text}")
            
            result = await response.json()
            return CodeResponse(
                code=result["choices"][0]["message"]["content"],
                raw_response=result
            )

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "provider": "DeepSeek",
            "context_length": 16384
        }

class TogetherProvider(BaseProvider):
    """Provider for Together.ai models (Code Llama, DeepSeek, StarCoder)"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "codellama/CodeLlama-34b-Python-hf"):
        super().__init__(api_key or os.getenv("TOGETHER_API_KEY"), model_name)
        together.api_key = self.api_key

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_code(self, prompt: str) -> CodeResponse:
        """Generate code using the Together API"""
        response = together.Complete.create(
            prompt=prompt,
            model=self.model_name,
            max_tokens=2048,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1.1
        )
        
        return CodeResponse(
            code=response["output"]["choices"][0]["text"],
            raw_response=response
        )

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "provider": "Together.ai",
            "context_length": 16384
        }

class AnthropicProvider(BaseProvider):
    """Provider for Anthropic models"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-opus-20240229"):
        super().__init__(api_key or os.getenv("ANTHROPIC_API_KEY"), model_name)
        self.client = anthropic.Anthropic(api_key=self.api_key)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_code(self, prompt: str) -> CodeResponse:
        """Generate code using the Anthropic API"""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=2048,
            temperature=0.7,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return CodeResponse(
            code=response.content[0].text,
            raw_response=response
        )

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "provider": "Anthropic",
            "context_length": 200000
        }

class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gpt-4-0125-preview"):
        super().__init__(api_key or os.getenv("OPENAI_API_KEY"), model_name)
        self.client = openai.OpenAI(api_key=self.api_key)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_code(self, prompt: str) -> CodeResponse:
        """Generate code using the OpenAI API"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2048
        )
        
        return CodeResponse(
            code=response.choices[0].message.content,
            raw_response=response
        )

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "provider": "OpenAI",
            "context_length": 128000
        }

class GeminiProvider(BaseProvider):
    """Provider for Google's Gemini models"""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
        super().__init__(api_key or os.getenv("GOOGLE_API_KEY"), model_name)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def generate_code(self, prompt: str) -> CodeResponse:
        """Generate code using the Gemini API"""
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.7,
                max_output_tokens=2048,
            )
        )
        
        return CodeResponse(
            code=response.text,
            raw_response=response
        )

    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "name": self.model_name,
            "provider": "Google",
            "context_length": 32768
        }

def get_provider(provider_name: str, api_key: Optional[str] = None, model_name: Optional[str] = None) -> BaseProvider:
    """Factory function to get the appropriate provider instance"""
    providers = {
        "deepseek": DeepSeekProvider,
        "together": TogetherProvider,
        "anthropic": AnthropicProvider,
        "openai": OpenAIProvider,
        "gemini": GeminiProvider
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}")
        
    provider_class = providers[provider_name]
    return provider_class(api_key=api_key, model_name=model_name) if model_name else provider_class(api_key=api_key) 