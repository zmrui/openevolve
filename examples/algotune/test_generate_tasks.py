#!/usr/bin/env python3
"""
Test script to generate and validate a few AlgoTune tasks.

This script processes only the first 5 tasks to test the validation logic.
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
                
                if not syntax_ok:
                    results['overall_valid'] = False
                    print(f"  ❌ {file_name}: {syntax_msg}")
                elif not imports_ok:
                    results['overall_valid'] = False
                    print(f"  ⚠️  {file_name}: {imports_msg}")
                else:
                    print(f"  ✅ {file_name}: {syntax_msg}, {imports_msg}")
                    
            elif file_name.endswith('.yaml'):
                syntax_ok, syntax_msg = validate_yaml_syntax(file_path)
                results['syntax_valid'][file_name] = syntax_ok
                
                if not syntax_ok:
                    results['overall_valid'] = False
                    print(f"  ❌ {file_name}: {syntax_msg}")
                else:
                    print(f"  ✅ {file_name}: {syntax_msg}")
        else:
            results['overall_valid'] = False
            print(f"  ❌ {file_name}: File missing")
    
    return results

def main():
    """Main function to generate and validate a few tasks."""
    print("🚀 Testing task generation and validation...")
    print("=" * 60)
    
    # Initialize the task adapter
    adapter = AlgoTuneTaskAdapter()
    available_tasks = adapter.list_available_tasks()
    
    # Only process the first 5 tasks for testing
    test_tasks = available_tasks[:5]
    
    print(f"Testing with {len(test_tasks)} tasks:")
    for task in test_tasks:
        print(f"  - {task}")
    print()
    
    # Track results
    successful_tasks = []
    failed_tasks = []
    validation_results = {}
    
    # Generate tasks
    print("📝 Generating OpenEvolve files for each task...")
    for i, task_name in enumerate(test_tasks, 1):
        print(f"\n[{i}/{len(test_tasks)}] Processing {task_name}...")
        
        try:
            # Generate the task files
            output_path = adapter.create_task_files(task_name)
            print(f"  ✅ Generated files in: {output_path}")
            
            # Validate the generated structure
            task_dir = Path(output_path)
            validation_result = validate_task_structure(task_dir)
            validation_results[task_name] = validation_result
            
            if validation_result['overall_valid']:
                successful_tasks.append(task_name)
                print(f"  ✅ {task_name}: All validations passed")
            else:
                failed_tasks.append(task_name)
                print(f"  ❌ {task_name}: Some validations failed")
                
        except Exception as e:
            failed_tasks.append(task_name)
            print(f"  ❌ {task_name}: Error during generation - {e}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total tasks processed: {len(test_tasks)}")
    print(f"Successful generations: {len(successful_tasks)}")
    print(f"Failed generations: {len(failed_tasks)}")
    
    if successful_tasks:
        print(f"\n✅ Successfully generated tasks:")
        for task in successful_tasks:
            print(f"  - {task}")
    
    if failed_tasks:
        print(f"\n❌ Failed tasks:")
        for task in failed_tasks:
            print(f"  - {task}")
    
    # Detailed validation report
    print("\n" + "=" * 60)
    print("🔍 DETAILED VALIDATION REPORT")
    print("=" * 60)
    
    for task_name, result in validation_results.items():
        print(f"\n📁 {task_name}:")
        
        # Check file presence
        for file_name, present in result['files_present'].items():
            status = "✅" if present else "❌"
            print(f"  {status} {file_name}: {'Present' if present else 'Missing'}")
        
        # Check syntax and imports
        for file_name in ['initial_program.py', 'evaluator.py']:
            if file_name in result['syntax_valid']:
                syntax_status = "✅" if result['syntax_valid'][file_name] else "❌"
                imports_status = "✅" if result['imports_valid'].get(file_name, True) else "⚠️"
                print(f"  {syntax_status} {file_name} syntax")
                print(f"  {imports_status} {file_name} imports")
        
        if 'config.yaml' in result['syntax_valid']:
            yaml_status = "✅" if result['syntax_valid']['config.yaml'] else "❌"
            print(f"  {yaml_status} config.yaml syntax")
    
    # Overall success rate
    success_rate = len(successful_tasks) / len(test_tasks) * 100
    print(f"\n📈 Test Success Rate: {success_rate:.1f}%")
    
    if failed_tasks:
        print(f"\n💡 Recommendations:")
        print("  - Check the failed tasks for specific error messages")
        print("  - Verify that all AlgoTune tasks have proper structure")
        print("  - Ensure all tasks have solve methods with correct indentation")
        return 1
    else:
        print(f"\n🎉 All test tasks generated successfully!")
        print("Ready to run the full script with all 155 tasks.")
        return 0

if __name__ == "__main__":
    sys.exit(main()) 