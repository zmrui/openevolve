"""
Evaluator for HuggingFace dataset-based prompt optimization.
"""

import re
import traceback
import yaml
import os
import time
from openai import OpenAI
from tqdm import tqdm
from datasets import load_dataset

# Read config.yaml to get model settings
with open(os.path.join(os.path.dirname(__file__), "config.yaml"), 'r') as f:
    config = yaml.safe_load(f)

# Get model settings from config
llm_config = config.get('llm', {})
api_base = llm_config.get('api_base', 'http://localhost:1234/v1')

# Handle both single model and model list configurations
models = llm_config.get('models', [])
if models:
    # Use first model from list
    TASK_MODEL_NAME = models[0].get('name', 'default-model')
else:
    # Fallback to direct model specification
    TASK_MODEL_NAME = llm_config.get('primary_model', 'default-model')

# Get evaluator settings
evaluator_config = config.get('evaluator', {})
MAX_RETRIES = evaluator_config.get('max_retries', 3)

# Initialize OpenAI client once for all evaluations
test_model = OpenAI(base_url=api_base)
print(f"Initialized OpenAI client with model: {TASK_MODEL_NAME}")

def load_prompt_config(prompt_path):
    """Load the prompt from text file and dataset config from dataset.yaml."""
    # Load prompt from text file
    with open(prompt_path, 'r') as f:
        prompt = f.read().strip()
    
    # Always load dataset configuration from the examples directory
    # This ensures it works even when OpenEvolve copies files to temp directories
    evaluator_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(evaluator_dir, 'dataset.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config, prompt

def load_hf_dataset(config):
    """Load HuggingFace dataset based on configuration."""
    dataset_name = config['dataset_name']
    split = config.get('split', 'test')
    
    print(f"Loading dataset: {dataset_name}")
    
    try:
        # Try to load the specified split
        dataset = load_dataset(dataset_name, split=split)
    except:
        # Fallback to train split if test is not available
        print(f"Split '{split}' not found, falling back to 'train'")
        dataset = load_dataset(dataset_name, split='train')
    
    print(f"Dataset loaded with {len(dataset)} examples")
    return dataset

def evaluate_prompt(prompt, dataset, config, num_samples):
    """Evaluate a prompt on a subset of the dataset."""
    input_field = config['input_field']
    target_field = config['target_field']
    
    # Sample from dataset
    samples = dataset.select(range(min(num_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for example in tqdm(samples, desc=f"Evaluating {num_samples} samples"):
        input_text = example[input_field]
        expected = example[target_field]
        
        # Prepare the message for the LLM
        messages = [
            {"role": "user", "content": prompt.format(input_text=input_text)}
        ]
        
        # Call the LLM with retry logic
        for attempt in range(MAX_RETRIES):
            try:
                response = test_model.chat.completions.create(
                    model=TASK_MODEL_NAME,
                    messages=messages,
                    temperature=0.1,  # Low temperature for consistent classification
                    max_tokens=10  # We only need a short response
                )
                break
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    print(f"Failed to get response after {MAX_RETRIES} attempts: {e}")
                    raise e
                time.sleep(1)
        
        # Handle potential None response
        if not response:
            print(f"Warning: No response object from LLM")
            total += 1  # Count as incorrect
            continue
            
        if not response.choices:
            print(f"Warning: No choices in response from LLM")
            total += 1  # Count as incorrect
            continue
            
        if not response.choices[0].message:
            print(f"Warning: No message in response choice")
            total += 1  # Count as incorrect
            continue
            
        output_text = response.choices[0].message.content
        if output_text is None:
            print(f"Warning: None content in LLM response")
            print(f"Full response: {response}")
            total += 1  # Count as incorrect
            continue
            
        output_text = output_text.strip()
        
        # Extract prediction from output
        try:
            # Look for a number (0 or 1) in the output
            numbers = re.findall(r'\b[01]\b', output_text)
            if numbers:
                prediction = int(numbers[-1])  # Use the last number found
            else:
                # Try to infer from keywords
                output_lower = output_text.lower()
                if 'positive' in output_lower:
                    prediction = 1
                elif 'negative' in output_lower:
                    prediction = 0
                else:
                    prediction = -1  # Invalid prediction
            
            if prediction == expected:
                correct += 1
            
            total += 1
            
        except Exception as e:
            print(f"Error parsing response '{output_text}': {e}")
            total += 1  # Count as incorrect
    
    accuracy = correct / total if total > 0 else 0.0
    return accuracy, correct, total

def evaluate_stage1(prompt_path):
    """
    Stage 1 evaluation: Quick evaluation with 10% of samples
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        Dictionary with combined_score metric
    """
    print('-' * 80)
    print("Starting Stage 1 evaluation...")
    print('-' * 80)
    
    try:
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")
        
        # Load dataset
        dataset = load_hf_dataset(config)
        
        # Get number of samples from config
        num_samples = config.get('max_samples', 50)
        stage1_samples = max(10, int(num_samples * 0.1))
        
        print(f"Stage 1: Evaluating {stage1_samples} samples...")
        
        # Run evaluation
        accuracy, correct, total = evaluate_prompt(
            prompt, dataset, config, stage1_samples
        )
        
        print(f"Stage 1 accuracy: {accuracy:.3f} ({correct}/{total})")
        print('-' * 80)
        
        return {
            "combined_score": accuracy
        }
        
    except Exception as e:
        print(f"Stage 1 evaluation failed: {str(e)}")
        traceback.print_exc()
        print('-' * 80)
        return {
            "combined_score": 0.0,
            "error": str(e)
        }


def evaluate_stage2(prompt_path):
    """
    Stage 2 evaluation: Full evaluation with all samples
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        Dictionary with combined_score metric
    """
    print('-' * 80)
    print("Starting Stage 2 evaluation...")
    print('-' * 80)
    
    try:
        # Load prompt configuration
        config, prompt = load_prompt_config(prompt_path)
        print(f"Loaded prompt configuration")
        
        # Load dataset
        dataset = load_hf_dataset(config)
        
        # Get number of samples from config
        num_samples = config.get('max_samples', 50)
        
        print(f"Stage 2: Evaluating all {num_samples} samples...")
        
        # Run evaluation
        accuracy, correct, total = evaluate_prompt(
            prompt, dataset, config, num_samples
        )
        
        print(f"Stage 2 accuracy: {accuracy:.3f} ({correct}/{total})")
        print('-' * 80)
        
        return {
            "combined_score": accuracy
        }
        
    except Exception as e:
        print(f"Stage 2 evaluation failed: {str(e)}")
        traceback.print_exc()
        print('-' * 80)
        return {
            "combined_score": 0.0,
            "error": str(e)
        }


def evaluate(prompt_path):
    """
    Main evaluation function - for backwards compatibility
    Calls evaluate_stage2 for full evaluation
    
    Args:
        prompt_path: Path to the prompt file
        
    Returns:
        Dictionary with combined_score metric
    """
    return evaluate_stage2(prompt_path)