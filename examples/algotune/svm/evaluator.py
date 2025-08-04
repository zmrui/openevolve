"""
Evaluator for the svm task
"""

import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import logging
import sys
import os
from pathlib import Path

# Add AlgoTune to path for importing reference tasks
LOCAL_ALGOTUNE_PATH = Path(__file__).parent.parent / "AlgoTuneTasks"
LOCAL_ALGOTUNER_PATH = Path(__file__).parent.parent / "AlgoTuner"

# Try local paths first, then fallback to parent directory
if LOCAL_ALGOTUNER_PATH.exists():
    sys.path.insert(0, str(LOCAL_ALGOTUNER_PATH.parent))
elif LOCAL_ALGOTUNE_PATH.exists():
    sys.path.insert(0, str(LOCAL_ALGOTUNE_PATH.parent))

# Try to import AlgoTune tasks
try:
    from AlgoTuneTasks.base import TASK_REGISTRY
    # Import the specific svm task to register it
    from AlgoTuneTasks.svm.svm import SVMTask
    print("Successfully imported AlgoTune tasks and svm")
except ImportError as e:
    print(f"Error: Could not import AlgoTune tasks: {e}")
    print("Make sure AlgoTune is properly installed and accessible")
    TASK_REGISTRY = {}

def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=30):
    """
    Run a function with a timeout using concurrent.futures

    Args:
        func: Function to run
        args: Arguments to pass to the function
        kwargs: Keyword arguments to pass to the function
        timeout_seconds: Timeout in seconds

    Returns:
        Result of the function or raises TimeoutError
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, *args, **kwargs)
        try:
            result = future.result(timeout=timeout_seconds)
            return result
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"Function timed out after {timeout_seconds} seconds")

def safe_convert(value):
    """Convert a value safely for evaluation"""
    try:
        if isinstance(value, (list, tuple)):
            return [safe_convert(v) for v in value]
        elif isinstance(value, np.ndarray):
            return value.tolist()
        else:
            return value
    except Exception:
        return value

def evaluate(program_path, config=None):
    """
    Evaluate the evolved program by running it on test problems and comparing
    with reference solutions from the original AlgoTune task.
    
    This evaluator:
    1. Loads the evolved solve method from initial_program.py
    2. Generates test problems using the original AlgoTune task
    3. Runs the evolved solution and measures performance
    4. Validates correctness using the original task's validation method

    Args:
        program_path: Path to the evolved program file (initial_program.py)
        config: Configuration dictionary with evaluator settings

    Returns:
        Dictionary of metrics
    """
    try:
        # Load configuration
        if config is None:
            # Default configuration if none provided
            config = {
                "algotune": {
                    "num_trials": 5,
                    "data_size": 5,
                    "timeout": 30
                }
            }
        
        # Extract AlgoTune task-specific settings from config
        algotune_config = config.get("algotune", {})
        num_trials = algotune_config.get("num_trials", 5)
        data_size = algotune_config.get("data_size", 5)
        timeout_seconds = algotune_config.get("timeout", 30)
        
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        # The run_solver function calls the evolved solve method from the class
        if not hasattr(program, "run_solver"):
            print(f"Error: program does not have 'run_solver' function")
            return {
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing run_solver function",
            }

        # Get the original task for reference solutions and problem generation
        task_class = None
        if "svm" in TASK_REGISTRY:
            task_class = TASK_REGISTRY["svm"]
            print(f"Successfully loaded svm task from registry")
        else:
            print(f"Error: svm task not found in TASK_REGISTRY")
            print(f"Available tasks: {list(TASK_REGISTRY.keys())}")
            raise Exception("Could not load svm task from AlgoTune registry")

        # Generate test problems
        correctness_scores = []
        performance_scores = []
        success_count = 0

        for trial in range(num_trials):
            try:
                start_time = time.time()

                # Generate a test problem using the original task
                if task_class:
                    # Use the original task to generate problems
                    task_instance = task_class()
                    problem = task_instance.generate_problem(n=data_size, random_seed=trial)
                else:
                    raise Exception("Could not load original AlgoTune task for problem generation")

                # Run the evolved solution from initial_program.py
                result = run_with_timeout(program.run_solver, args=(problem,), timeout_seconds=timeout_seconds)
                end_time = time.time()

                # Convert result to comparable format
                result = safe_convert(result)

                # Evaluate correctness using the evolved solve method
                correctness_score = 0.0
                if task_class:
                    try:
                        # Check if solution is valid using original task's is_solution method
                        is_valid = task_instance.is_solution(problem, result)
                        correctness_score = 1.0 if is_valid else 0.0
                    except Exception as e:
                        print(f"Trial {trial}: Error checking solution validity: {e}")
                        correctness_score = 0.0
                else:
                    raise Exception("Could not load original AlgoTune task for solution validation")

                # Evaluate performance (time-based)
                execution_time = end_time - start_time
                performance_score = 1.0 / (1.0 + execution_time) if execution_time > 0 else 0.0

                correctness_scores.append(correctness_score)
                performance_scores.append(performance_score)
                success_count += 1

            except TimeoutError as e:
                print(f"Trial {trial}: {str(e)}")
                continue
            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                print(traceback.format_exc())
                continue

        # If all trials failed, return zero scores
        if success_count == 0:
            return {
                "correctness_score": 0.0,
                "performance_score": 0.0,
                "combined_score": 0.0,
                "error": "All trials failed",
            }

        # Calculate metrics
        avg_correctness = float(np.mean(correctness_scores))
        avg_performance = float(np.mean(performance_scores))
        reliability_score = float(success_count / num_trials)

        # Combined score prioritizing correctness
        combined_score = float(
            0.7 * avg_correctness + 0.2 * avg_performance + 0.1 * reliability_score
        )

        return {
            "correctness_score": avg_correctness,
            "performance_score": avg_performance,
            "reliability_score": reliability_score,
            "combined_score": combined_score,
            "success_rate": reliability_score,
        }

    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "correctness_score": 0.0,
            "performance_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
        }

# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path, config=None):
    """First stage evaluation with basic functionality check of the evolved solve method"""
    try:
        # Load configuration
        if config is None:
            config = {
                "algotune": {
                    "num_trials": 5,
                    "data_size": 5,
                    "timeout": 30
                }
            }
        
        algotune_config = config.get("algotune", {})
        data_size = algotune_config.get("data_size", 5)
        timeout_seconds = algotune_config.get("timeout", 30)
        
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_solver"):
            return {"runs_successfully": 0.0, "error": "Missing run_solver function"}

        # Get the original task for reference solutions and problem generation
        task_class = None
        if "svm" in TASK_REGISTRY:
            task_class = TASK_REGISTRY["svm"]
        else:
            print(f"Error: svm task not found in TASK_REGISTRY")
            print(f"Available tasks: {list(TASK_REGISTRY.keys())}")

        try:
            # Run a single trial with timeout using proper task-specific problem
            if task_class:
                task_instance = task_class()
                test_problem = task_instance.generate_problem(n=data_size, random_seed=42)
            else:
                # Generic fallback test problem
                test_problem = {"test_data": [1, 2, 3], "random_seed": 42}
            
            result = run_with_timeout(program.run_solver, args=(test_problem,), timeout_seconds=timeout_seconds)

            # Basic validity check
            if result is not None:
                return {
                    "runs_successfully": 1.0,
                    "basic_functionality": 1.0,
                }
            else:
                return {
                    "runs_successfully": 0.5,
                    "basic_functionality": 0.0,
                    "error": "Function returned None"
                }

        except TimeoutError as e:
            return {"runs_successfully": 0.0, "error": "Timeout"}
        except Exception as e:
            return {"runs_successfully": 0.0, "error": str(e)}

    except Exception as e:
        return {"runs_successfully": 0.0, "error": str(e)}

def evaluate_stage2(program_path, config=None):
    """Second stage evaluation with more thorough testing of the evolved solve method"""
    return evaluate(program_path, config)
