"""
Test script for news transformation evaluation
"""

import asyncio
from news_style_evaluation import DeepSeekProvider, NewsStyleEvaluator

async def main():
    # DeepSeek API key
    api_key = "sk-da86eedf815d42859159f9f8e94a8ee4"
    
    print("Starting news transformation evaluation with DeepSeek...")
    
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
            print(f"Evaluation Time: {results['evaluation_timestamp']}")
            print("\nTransformations:")
            
            for i, transform in enumerate(results['transformations'], 1):
                print(f"\nArticle {i}:")
                print(f"Original Title: {transform['original_title']}")
                print(f"URL: {transform.get('url', 'N/A')}")
                print(f"Published: {transform.get('published_date', 'N/A')}")
                print(f"Score: {transform['score']:.2f}")
                print("\nTransformed Content:")
                print("-" * 30)
                print(transform['transformed_content'])
                print("-" * 30)
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 