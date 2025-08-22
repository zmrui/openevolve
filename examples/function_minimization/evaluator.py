"""
Evaluator for the function minimization example
"""

import importlib.util
import numpy as np
import time
import concurrent.futures
import traceback
import signal


def run_with_timeout(func, args=(), kwargs={}, timeout_seconds=5):
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


def safe_float(value):
    """Convert a value to float safely"""
    try:
        return float(value)
    except (TypeError, ValueError):
        print(f"Warning: Could not convert {value} of type {type(value)} to float")
        return 0.0


def evaluate(program_path):
    """
    Evaluate the program by running it multiple times and checking how close
    it gets to the known global minimum.

    Args:
        program_path: Path to the program file

    Returns:
        Dictionary of metrics
    """
    # Known global minimum (approximate)
    GLOBAL_MIN_X = -1.704
    GLOBAL_MIN_Y = 0.678
    GLOBAL_MIN_VALUE = -1.519

    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_search"):
            print(f"Error: program does not have 'run_search' function")
            return {
                "value_score": 0.0,
                "distance_score": 0.0,
                "reliability_score": 0.0,
                "combined_score": 0.0,
                "error": "Missing run_search function",
            }

        # Run multiple trials
        num_trials = 10
        x_values = []
        y_values = []
        values = []
        distances = []
        times = []
        success_count = 0

        for trial in range(num_trials):
            try:
                start_time = time.time()

                # Run with timeout
                result = run_with_timeout(program.run_search, timeout_seconds=5)

                # Handle different result formats
                if isinstance(result, tuple):
                    if len(result) == 3:
                        x, y, value = result
                    elif len(result) == 2:
                        # Assume it's (x, y) and calculate value
                        x, y = result
                        # Calculate the function value since it wasn't returned
                        value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                        print(f"Trial {trial}: Got 2 values, calculated function value: {value}")
                    else:
                        print(
                            f"Trial {trial}: Invalid result format, expected tuple of 2 or 3 values but got {len(result)}"
                        )
                        continue
                else:
                    print(
                        f"Trial {trial}: Invalid result format, expected tuple but got {type(result)}"
                    )
                    continue

                end_time = time.time()

                # Ensure all values are float
                x = safe_float(x)
                y = safe_float(y)
                value = safe_float(value)

                # Check if the result is valid (not NaN or infinite)
                if (
                    np.isnan(x)
                    or np.isnan(y)
                    or np.isnan(value)
                    or np.isinf(x)
                    or np.isinf(y)
                    or np.isinf(value)
                ):
                    print(f"Trial {trial}: Invalid result, got x={x}, y={y}, value={value}")
                    continue

                # Calculate metrics
                x_diff = x - GLOBAL_MIN_X
                y_diff = y - GLOBAL_MIN_Y
                distance_to_global = np.sqrt(x_diff**2 + y_diff**2)

                x_values.append(x)
                y_values.append(y)
                values.append(value)
                distances.append(distance_to_global)
                times.append(end_time - start_time)
                success_count += 1

            except TimeoutError as e:
                print(f"Trial {trial}: {str(e)}")
                continue
            except IndexError as e:
                # Specifically handle IndexError which often happens with early termination checks
                print(f"Trial {trial}: IndexError - {str(e)}")
                print(
                    "This is likely due to a list index check before the list is fully populated."
                )
                continue
            except Exception as e:
                print(f"Trial {trial}: Error - {str(e)}")
                print(traceback.format_exc())
                continue

        # If all trials failed, return zero scores
        if success_count == 0:
            return {
                "value_score": 0.0,
                "distance_score": 0.0,
                "reliability_score": 0.0,
                "combined_score": 0.0,
                "error": "All trials failed",
            }

        # Calculate metrics
        avg_value = float(np.mean(values))
        avg_distance = float(np.mean(distances))
        avg_time = float(np.mean(times)) if times else 1.0

        # Convert to scores (higher is better)
        value_score = float(1.0 / (1.0 + abs(avg_value - GLOBAL_MIN_VALUE)))
        distance_score = float(1.0 / (1.0 + avg_distance))
        
        # Add reliability score based on success rate
        reliability_score = float(success_count / num_trials)

        # Calculate solution quality based on distance to global minimum
        if avg_distance < 0.5:  # Very close to the correct solution
            solution_quality_multiplier = 1.5  # 50% bonus
        elif avg_distance < 1.5:  # In the right region
            solution_quality_multiplier = 1.2  # 20% bonus
        elif avg_distance < 3.0:  # Getting closer
            solution_quality_multiplier = 1.0  # No adjustment
        else:  # Not finding the right region
            solution_quality_multiplier = 0.7  # 30% penalty

        # Calculate combined score that prioritizes finding the global minimum
        # Base score from value and distance, then apply solution quality multiplier
        base_score = 0.5 * value_score + 0.3 * distance_score + 0.2 * reliability_score
        combined_score = float(base_score * solution_quality_multiplier)

        return {
            "value_score": value_score,
            "distance_score": distance_score,
            "reliability_score": reliability_score,
            "combined_score": combined_score,
        }
    except Exception as e:
        print(f"Evaluation failed completely: {str(e)}")
        print(traceback.format_exc())
        return {
            "value_score": 0.0,
            "distance_score": 0.0,
            "reliability_score": 0.0,
            "combined_score": 0.0,
            "error": str(e),
        }


# Stage-based evaluation for cascade evaluation
def evaluate_stage1(program_path):
    """First stage evaluation with fewer trials"""
    # Known global minimum (approximate)
    GLOBAL_MIN_X = float(-1.704)
    GLOBAL_MIN_Y = float(0.678)
    GLOBAL_MIN_VALUE = float(-1.519)

    # Quick check to see if the program runs without errors
    try:
        # Load the program
        spec = importlib.util.spec_from_file_location("program", program_path)
        program = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(program)

        # Check if the required function exists
        if not hasattr(program, "run_search"):
            print(f"Stage 1 validation: Program does not have 'run_search' function")
            return {
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "error": "Missing run_search function"
            }

        try:
            # Run a single trial with timeout
            result = run_with_timeout(program.run_search, timeout_seconds=5)

            # Handle different result formats
            if isinstance(result, tuple):
                if len(result) == 3:
                    x, y, value = result
                elif len(result) == 2:
                    # Assume it's (x, y) and calculate value
                    x, y = result
                    # Calculate the function value since it wasn't returned
                    value = np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20
                    print(f"Stage 1: Got 2 values, calculated function value: {value}")
                else:
                    print(
                        f"Stage 1: Invalid result format, expected tuple of 2 or 3 values but got {len(result)}"
                    )
                    return {
                        "runs_successfully": 0.0, 
                        "combined_score": 0.0,
                        "error": "Invalid result format"
                    }
            else:
                print(f"Stage 1: Invalid result format, expected tuple but got {type(result)}")
                return {
                    "runs_successfully": 0.0, 
                    "combined_score": 0.0,
                    "error": "Invalid result format"
                }

            # Ensure all values are float
            x = safe_float(x)
            y = safe_float(y)
            value = safe_float(value)

            # Check if the result is valid
            if (
                np.isnan(x)
                or np.isnan(y)
                or np.isnan(value)
                or np.isinf(x)
                or np.isinf(y)
                or np.isinf(value)
            ):
                print(f"Stage 1 validation: Invalid result, got x={x}, y={y}, value={value}")
                return {
                    "runs_successfully": 0.5, 
                    "combined_score": 0.0,
                    "error": "Invalid result values"
                }

            # Calculate distance safely
            x_diff = float(x) - GLOBAL_MIN_X
            y_diff = float(y) - GLOBAL_MIN_Y
            distance = float(np.sqrt(x_diff**2 + y_diff**2))

            # Calculate value-based score
            value_score = float(1.0 / (1.0 + abs(value - GLOBAL_MIN_VALUE)))
            distance_score = float(1.0 / (1.0 + distance))

            # Calculate solution quality based on distance to global minimum
            if distance < 0.5:  # Very close to the correct solution
                solution_quality_multiplier = 1.4  # 40% bonus
            elif distance < 1.5:  # In the right region
                solution_quality_multiplier = 1.15  # 15% bonus
            elif distance < 3.0:  # Getting closer
                solution_quality_multiplier = 1.0  # No adjustment
            else:  # Not finding the right region
                solution_quality_multiplier = 0.8  # 20% penalty

            # Calculate combined score for stage 1
            base_score = 0.6 * value_score + 0.4 * distance_score
            combined_score = float(base_score * solution_quality_multiplier)

            return {
                "runs_successfully": 1.0,
                "value_score": value_score,
                "distance_score": distance_score,
                "combined_score": combined_score,
            }
        except TimeoutError as e:
            print(f"Stage 1 evaluation timed out: {e}")
            return {
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "error": "Timeout"
            }
        except IndexError as e:
            # Specifically handle IndexError which often happens with early termination checks
            print(f"Stage 1 evaluation failed with IndexError: {e}")
            print("This is likely due to a list index check before the list is fully populated.")
            return {
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "error": f"IndexError: {str(e)}"
            }
        except Exception as e:
            print(f"Stage 1 evaluation failed: {e}")
            print(traceback.format_exc())
            return {
                "runs_successfully": 0.0, 
                "combined_score": 0.0,
                "error": str(e)
            }

    except Exception as e:
        print(f"Stage 1 evaluation failed: {e}")
        print(traceback.format_exc())
        return {
            "runs_successfully": 0.0, 
            "combined_score": 0.0,
            "error": str(e)
        }


def evaluate_stage2(program_path):
    """Second stage evaluation with more thorough testing"""
    # Full evaluation as in the main evaluate function
    return evaluate(program_path)
