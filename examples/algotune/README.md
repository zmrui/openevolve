# AlgoTune Task Adapter for OpenEvolve

This directory contains tools to evaluate OpenEvolve on AlgoTune tasks.

## Overview

The AlgoTune Task Adapter automatically extracts and converts AlgoTune tasks into OpenEvolve-compatible format. This adapter:

- **Extracts task information** from AlgoTune's task registry and description files
- **Generates OpenEvolve files** including initial programs, evaluators, and configurations
- **Preserves task semantics** while adapting to OpenEvolve's evolution framework
- **Supports all 155 algorithmic tasks** from the AlgoTune benchmark suite

### Key Contributions

The adapter performs several critical transformations:

1. **Task Extraction**: Parses AlgoTune task files to extract class definitions, solve methods, and problem specifications
2. **Code Generation**: Creates OpenEvolve-compatible initial programs with proper evolution blocks
3. **Evaluation Integration**: Generates evaluators that compare evolved solutions against original AlgoTune reference implementations
4. **Configuration Mapping**: Translates AlgoTune task parameters to OpenEvolve configuration settings

## Generated Files

For each task, the adapter creates:
- `initial_program.py` - The starting program for evolution with EVOLVE-BLOCK markers
- `evaluator.py` - The evaluation function that tests correctness and performance
- `config.yaml` - OpenEvolve configuration with task-specific settings

## Example: SVM Task

Here's an example of how the adapter transforms an AlgoTune SVM task:

### Generated Initial Program (`initial_program.py`)

```python
# EVOLVE-BLOCK-START
"""
SVM Task
Given labels y ∈ {-1, 1}^n and a feature matrix X ∈ R^{n x p} with rows x_1,...,x_n, 
solve the support vector machine (SVM) task

min        1/2 || β ||_2^2 + C sum_{i=1}^n ξ_i
β,β_0,ξ  

subject to ξ_i ≥ 0, i = 1,...,n
	   y_i (x_i^T β + β_0) ≥ 1 - ξ_i, i = 1,...,n

Input: Dictionary with keys "X", "y", "C"
Output: Dictionary with keys "beta0", "beta", "optimal_value", "misclass_error"
"""
import cvxpy as cp
import numpy as np

class SVMTask:
    def solve(self, problem):
        """Solve the SVM problem using CVXPY."""
        X = np.array(problem["X"])
        y = np.array(problem["y"])[:, None]
        C = float(problem["C"])
        
        p, n = X.shape[1], X.shape[0]
        beta = cp.Variable((p, 1))
        beta0 = cp.Variable()
        xi = cp.Variable((n, 1))
        
        objective = cp.Minimize(0.5 * cp.sum_squares(beta) + C * cp.sum(xi))
        constraints = [
            xi >= 0,
            cp.multiply(y, X @ beta + beta0) >= 1 - xi,
        ]
        
        problem_cp = cp.Problem(objective, constraints)
        problem_cp.solve()
        
        return {
            "beta0": float(beta0.value),
            "beta": beta.value.flatten().tolist(),
            "optimal_value": float(problem_cp.value),
            "misclass_error": self._compute_misclassification_error(X, y, beta.value, beta0.value)
        }
# EVOLVE-BLOCK-END
```

### Generated Configuration (`config.yaml`)

```yaml
# Configuration for svm task
max_iterations: 100
checkpoint_interval: 10

# LLM configuration
llm:
  primary_model: "gpt-4o-mini"
  temperature: 0.7
  max_tokens: 4096

# Prompt configuration
prompt:
  system_message: "You are an expert programmer specializing in convex_optimization algorithms. Your task is to improve the svm algorithm implementation..."

# AlgoTune task-specific configuration
algotune:
  num_trials: 5
  data_size: 5
  timeout: 30
```

### Generated Evaluator (`evaluator.py`)

The evaluator:
- Loads the evolved solve method from `initial_program.py`
- Generates test problems using the original AlgoTune task
- Runs the evolved solution and measures performance
- Validates correctness using the original task's validation method
- Compares against reference solutions from AlgoTune

## Files

- `task_adapter.py` - Main adapter that converts AlgoTune tasks to OpenEvolve format
- `create_task.py` - Simple script to create OpenEvolve files for a single task
- `generate_all_tasks.py` - Script to generate all 155 AlgoTune tasks

## Usage

### Generate a Single Task

```bash
# List all available tasks
python create_task.py --list

# Generate OpenEvolve files for a specific task
python create_task.py svm
python create_task.py kmeans
```

### Generate All Tasks

```bash
# Generate all 155 tasks
python generate_all_tasks.py
```

### Run Evolution

```bash
# Navigate to a generated task directory
cd svm/

# Run OpenEvolve on the task
 python ../../../openevolve-run.py ./initial_program.py ./evaluator.py \
    --config ./config.yaml \
    --output openevolve_output/
```