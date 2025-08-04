#!/usr/bin/env python3
"""
Simple script to create OpenEvolve tasks from AlgoTune tasks.

Usage:
    python create_task.py <task_name>
    python create_task.py --list
"""

import sys
from pathlib import Path

# Add the current directory to path so we can import task_adapter
sys.path.insert(0, str(Path(__file__).parent))

from task_adapter import AlgoTuneTaskAdapter

def main():
    if len(sys.argv) < 2:
        print("Usage: python create_task.py <task_name>")
        print("       python create_task.py --list")
        return 1
    
    task_name = sys.argv[1]
    
    if task_name == "--list":
        adapter = AlgoTuneTaskAdapter()
        print("Available AlgoTune tasks:")
        for task_name in adapter.list_available_tasks():
            print(f"  - {task_name}")
        return 0
    
    try:
        adapter = AlgoTuneTaskAdapter()
        output_path = adapter.create_task_files(task_name)
        print(f"✅ Successfully created OpenEvolve files for '{task_name}' in: {output_path}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 