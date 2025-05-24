"""
Test script for LLM evaluation suite
"""

import asyncio
import os
from dotenv import load_dotenv
from llm_evaluation import DeepSeekProvider, EvaluationSuite, create_default_tasks, EvaluationTask

# Load environment variables
load_dotenv()

# Additional test task
def create_test_tasks():
    tasks = create_default_tasks()
    
    # Add a technical writing task
    tasks.append(
        EvaluationTask(
            name="technical_writing",
            description="Transform technical documentation into user-friendly content",
            prompt="""
            Transform this technical documentation into user-friendly content:

            Documentation: The recursive algorithm implements a depth-first traversal 
            of the binary tree structure, with O(n) time complexity and O(h) space 
            complexity, where h represents the height of the tree.
            """,
            expected_elements=[
                "simple language",
                "analogy",
                "example",
                "step-by-step",
                "explanation"
            ],
            scoring_criteria={
                "simple language": 20,
                "analogy": 20,
                "example": 20,
                "step-by-step": 20,
                "explanation": 20
            }
        )
    )
    
    return tasks

async def main():
    # Use DeepSeek API key
    api_key = "sk-da86eedf815d42859159f9f8e94a8ee4"
    
    print("Starting LLM evaluation with DeepSeek...")
    
    try:
        async with DeepSeekProvider(api_key, "deepseek-chat") as provider:
            suite = EvaluationSuite(provider)
            
            # Add test tasks
            for task in create_test_tasks():
                suite.add_task(task)
            
            # Run evaluation
            results = await suite.run_evaluation()
            
            # Print results
            print("\nEvaluation Results:")
            print("-" * 50)
            print(f"Model: {results['model_info']['model']}")
            print(f"Provider: {results['model_info']['provider']}")
            print(f"Overall Score: {results['total_score']:.2f}")
            print("\nTask Results:")
            
            for task_name, task_result in results['task_results'].items():
                print(f"\n{task_name}:")
                print(f"Score: {task_result['score']:.2f}")
                print("Response:")
                print("-" * 30)
                print(task_result['response'])
                print("-" * 30)
    
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main()) 