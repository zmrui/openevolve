#!/usr/bin/env python3
"""
Script to generate all AlgoTune tasks and validate their structure.

This script:
1. Lists all available AlgoTune tasks
2. Generates OpenEvolve files for each task
"""

import os
import sys
import subprocess
import ast
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add the current directory to path so we can import task_adapter
sys.path.insert(0, str(Path(__file__).parent))

from task_adapter import AlgoTuneTaskAdapter

def validate_python_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate Python syntax of a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        ast.parse(content)
        return True, "Syntax OK"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error reading file: {e}"

def validate_yaml_syntax(file_path: Path) -> Tuple[bool, str]:
    """Validate YAML syntax of a file."""
    try:
        import yaml
        with open(file_path, 'r') as f:
            yaml.safe_load(f)
        return True, "YAML syntax OK"
    except Exception as e:
        return False, f"YAML syntax error: {e}"

def check_required_imports(file_path: Path) -> Tuple[bool, str]:
    """Check if the file has required imports for OpenEvolve."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Different import requirements for different file types
        if 'initial_program.py' in str(file_path):
            # Initial program files need basic imports
            required_imports = ['import logging', 'import numpy', 'from typing']
        elif 'evaluator.py' in str(file_path):
            # Evaluator files have different requirements
            required_imports = ['import logging', 'import numpy']
        else:
            # Other files have minimal requirements
            required_imports = ['import logging']
        
        missing_imports = []
        
        for imp in required_imports:
            if imp not in content:
                missing_imports.append(imp)
        
        if missing_imports:
            return False, f"Missing imports: {', '.join(missing_imports)}"
        return True, "Required imports present"
    except Exception as e:
        return False, f"Error checking imports: {e}"

def validate_task_structure(task_dir: Path) -> Dict[str, Any]:
    """Validate the structure of a generated task."""
    results = {
        'task_name': task_dir.name,
        'files_present': {},
        'syntax_valid': {},
        'imports_valid': {},
        'overall_valid': True
    }
    
    required_files = ['initial_program.py', 'evaluator.py', 'config.yaml']
    
    for file_name in required_files:
        file_path = task_dir / file_name
        results['files_present'][file_name] = file_path.exists()
        
        if file_path.exists():
            # Check syntax
            if file_name.endswith('.py'):
                syntax_ok, syntax_msg = validate_python_syntax(file_path)
                results['syntax_valid'][file_name] = syntax_ok
                
                # Check imports for Python files
                imports_ok, imports_msg = check_required_imports(file_path)
                results['imports_valid'][file_name] = imports_ok
                
                if not syntax_ok or not imports_ok:
                    results['overall_valid'] = False
                    
            elif file_name.endswith('.yaml'):
                syntax_ok, syntax_msg = validate_yaml_syntax(file_path)
                results['syntax_valid'][file_name] = syntax_ok
                
                if not syntax_ok:
                    results['overall_valid'] = False
        else:
            results['overall_valid'] = False
    
    return results

def main():
    """Main function to generate and validate all tasks."""
    # Initialize the task adapter
    adapter = AlgoTuneTaskAdapter()
    available_tasks = adapter.list_available_tasks()
    
    # Track results
    successful_tasks = []
    failed_tasks = []
    
    # Generate tasks
    for task_name in available_tasks:
        try:
            # Generate the task files
            output_path = adapter.create_task_files(task_name)
            
            # Validate the generated structure
            task_dir = Path(output_path)
            validation_result = validate_task_structure(task_dir)
            
            if validation_result['overall_valid']:
                successful_tasks.append(task_name)
            else:
                failed_tasks.append(task_name)
                
        except Exception as e:
            failed_tasks.append(task_name)
    
    # Print simple success message
    print(f"✅ Successfully generated {len(successful_tasks)} tasks")
    
if __name__ == "__main__":
    sys.exit(main()) 