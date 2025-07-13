"""
Tests for cascade evaluation validation functionality in openevolve.evaluator
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from openevolve.config import Config
from openevolve.evaluator import Evaluator
from openevolve.evaluation_result import EvaluationResult


class TestCascadeValidation(unittest.TestCase):
    """Tests for cascade evaluation configuration validation"""

    def setUp(self):
        """Set up test evaluator with cascade validation"""
        self.config = Config()
        
        # Create temporary evaluator files for testing
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Clean up temporary files"""
        # Clean up temp files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def _create_evaluator_file(self, filename: str, content: str) -> str:
        """Helper to create temporary evaluator file"""
        file_path = os.path.join(self.temp_dir, filename)
        with open(file_path, 'w') as f:
            f.write(content)
        return file_path

    def test_cascade_validation_with_valid_evaluator(self):
        """Test cascade validation with evaluator that has cascade functions"""
        # Create evaluator with cascade functions
        evaluator_content = '''
def evaluate_stage1(program_path):
    return {"stage1_score": 0.5}

def evaluate_stage2(program_path):
    return {"stage2_score": 0.7}

def evaluate_stage3(program_path):
    return {"stage3_score": 0.9}

def evaluate(program_path):
    return {"final_score": 1.0}
'''
        evaluator_path = self._create_evaluator_file("valid_cascade.py", evaluator_content)
        
        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path
        
        # Should not raise warnings for valid cascade evaluator
        with patch('openevolve.evaluator.logger') as mock_logger:
            evaluator = Evaluator(self.config.evaluator, None)
            
            # Should not have called warning
            mock_logger.warning.assert_not_called()

    def test_cascade_validation_warning_for_missing_functions(self):
        """Test cascade validation warns when cascade functions are missing"""
        # Create evaluator without cascade functions
        evaluator_content = '''
def evaluate(program_path):
    return {"score": 0.5}
'''
        evaluator_path = self._create_evaluator_file("no_cascade.py", evaluator_content)
        
        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path
        
        # Should warn about missing cascade functions
        with patch('openevolve.evaluator.logger') as mock_logger:
            evaluator = Evaluator(self.config.evaluator, None)
            
            # Should have warned about missing stage functions
            mock_logger.warning.assert_called()
            warning_call = mock_logger.warning.call_args[0][0]
            self.assertIn("cascade_evaluation: true", warning_call)
            self.assertIn("evaluate_stage1", warning_call)

    def test_cascade_validation_partial_functions(self):
        """Test cascade validation with only some cascade functions"""
        # Create evaluator with only stage1
        evaluator_content = '''
def evaluate_stage1(program_path):
    return {"stage1_score": 0.5}

def evaluate(program_path):
    return {"score": 0.5}
'''
        evaluator_path = self._create_evaluator_file("partial_cascade.py", evaluator_content)
        
        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path
        
        # Should not warn since stage1 exists (minimum requirement)
        with patch('openevolve.evaluator.logger') as mock_logger:
            evaluator = Evaluator(self.config.evaluator, None)
            
            # Should not warn since stage1 exists
            mock_logger.warning.assert_not_called()

    def test_no_cascade_validation_when_disabled(self):
        """Test no validation when cascade evaluation is disabled"""
        # Create evaluator without cascade functions
        evaluator_content = '''
def evaluate(program_path):
    return {"score": 0.5}
'''
        evaluator_path = self._create_evaluator_file("no_cascade.py", evaluator_content)
        
        # Configure WITHOUT cascade evaluation
        self.config.evaluator.cascade_evaluation = False
        self.config.evaluator.evaluation_file = evaluator_path
        
        # Should not perform validation or warn
        with patch('openevolve.evaluator.logger') as mock_logger:
            evaluator = Evaluator(self.config.evaluator, None)
            
            # Should not warn when cascade evaluation is disabled
            mock_logger.warning.assert_not_called()

    def test_direct_evaluate_supports_evaluation_result(self):
        """Test that _direct_evaluate supports EvaluationResult returns"""
        # Create evaluator that returns EvaluationResult
        evaluator_content = '''
from openevolve.evaluation_result import EvaluationResult

def evaluate(program_path):
    return EvaluationResult(
        metrics={"score": 0.8, "accuracy": 0.9},
        artifacts={"debug_info": "test data"}
    )
'''
        evaluator_path = self._create_evaluator_file("result_evaluator.py", evaluator_content)
        
        self.config.evaluator.cascade_evaluation = False
        self.config.evaluator.evaluation_file = evaluator_path
        self.config.evaluator.timeout = 10
        
        evaluator = Evaluator(self.config.evaluator, None)
        
        # Create a dummy program file
        program_path = self._create_evaluator_file("test_program.py", "def test(): pass")
        
        # Mock the evaluation process
        with patch('openevolve.evaluator.run_external_evaluator') as mock_run:
            mock_run.return_value = EvaluationResult(
                metrics={"score": 0.8, "accuracy": 0.9},
                artifacts={"debug_info": "test data"}
            )
            
            # Should handle EvaluationResult without issues
            result = evaluator._direct_evaluate(program_path)
            
            # Should return the EvaluationResult as-is
            self.assertIsInstance(result, EvaluationResult)
            self.assertEqual(result.metrics["score"], 0.8)
            self.assertEqual(result.artifacts["debug_info"], "test data")

    def test_direct_evaluate_supports_dict_result(self):
        """Test that _direct_evaluate still supports dict returns"""
        # Create evaluator that returns dict
        evaluator_content = '''
def evaluate(program_path):
    return {"score": 0.7, "performance": 0.85}
'''
        evaluator_path = self._create_evaluator_file("dict_evaluator.py", evaluator_content)
        
        self.config.evaluator.cascade_evaluation = False
        self.config.evaluator.evaluation_file = evaluator_path
        self.config.evaluator.timeout = 10
        
        evaluator = Evaluator(self.config.evaluator, None)
        
        # Create a dummy program file
        program_path = self._create_evaluator_file("test_program.py", "def test(): pass")
        
        # Mock the evaluation process
        with patch('openevolve.evaluator.run_external_evaluator') as mock_run:
            mock_run.return_value = {"score": 0.7, "performance": 0.85}
            
            # Should handle dict result without issues
            result = evaluator._direct_evaluate(program_path)
            
            # Should return the dict as-is
            self.assertIsInstance(result, dict)
            self.assertEqual(result["score"], 0.7)
            self.assertEqual(result["performance"], 0.85)

    def test_cascade_validation_with_class_based_evaluator(self):
        """Test cascade validation with class-based evaluator"""
        # Create class-based evaluator
        evaluator_content = '''
class Evaluator:
    def evaluate_stage1(self, program_path):
        return {"stage1_score": 0.5}
    
    def evaluate(self, program_path):
        return {"score": 0.5}

# Module-level functions (what validation looks for)
def evaluate_stage1(program_path):
    evaluator = Evaluator()
    return evaluator.evaluate_stage1(program_path)

def evaluate(program_path):
    evaluator = Evaluator()
    return evaluator.evaluate(program_path)
'''
        evaluator_path = self._create_evaluator_file("class_cascade.py", evaluator_content)
        
        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path
        
        # Should not warn since module-level functions exist
        with patch('openevolve.evaluator.logger') as mock_logger:
            evaluator = Evaluator(self.config.evaluator, None)
            
            mock_logger.warning.assert_not_called()

    def test_cascade_validation_with_syntax_error(self):
        """Test cascade validation handles syntax errors gracefully"""
        # Create evaluator with syntax error
        evaluator_content = '''
def evaluate_stage1(program_path)  # Missing colon
    return {"stage1_score": 0.5}
'''
        evaluator_path = self._create_evaluator_file("syntax_error.py", evaluator_content)
        
        # Configure for cascade evaluation
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = evaluator_path
        
        # Should handle syntax error and still warn about cascade
        with patch('openevolve.evaluator.logger') as mock_logger:
            evaluator = Evaluator(self.config.evaluator, None)
            
            # Should have warned about missing functions (due to import failure)
            mock_logger.warning.assert_called()

    def test_cascade_validation_nonexistent_file(self):
        """Test cascade validation with nonexistent evaluator file"""
        # Configure with nonexistent file
        self.config.evaluator.cascade_evaluation = True
        self.config.evaluator.evaluation_file = "/nonexistent/path.py"
        
        # Should handle missing file gracefully
        with patch('openevolve.evaluator.logger') as mock_logger:
            evaluator = Evaluator(self.config.evaluator, None)
            
            # Should have warned about missing functions (due to import failure)
            mock_logger.warning.assert_called()

    def test_process_evaluation_result_with_artifacts(self):
        """Test that _process_evaluation_result handles artifacts correctly"""
        evaluator_path = self._create_evaluator_file("dummy.py", "def evaluate(p): pass")
        
        self.config.evaluator.evaluation_file = evaluator_path
        evaluator = Evaluator(self.config.evaluator, None)
        
        # Test with EvaluationResult containing artifacts
        eval_result = EvaluationResult(
            metrics={"score": 0.9},
            artifacts={"log": "test log", "data": [1, 2, 3]}
        )
        
        metrics, artifacts = evaluator._process_evaluation_result(eval_result)
        
        self.assertEqual(metrics, {"score": 0.9})
        self.assertEqual(artifacts, {"log": "test log", "data": [1, 2, 3]})

    def test_process_evaluation_result_with_dict(self):
        """Test that _process_evaluation_result handles dict results correctly"""
        evaluator_path = self._create_evaluator_file("dummy.py", "def evaluate(p): pass")
        
        self.config.evaluator.evaluation_file = evaluator_path
        evaluator = Evaluator(self.config.evaluator, None)
        
        # Test with dict result
        dict_result = {"score": 0.7, "accuracy": 0.8}
        
        metrics, artifacts = evaluator._process_evaluation_result(dict_result)
        
        self.assertEqual(metrics, {"score": 0.7, "accuracy": 0.8})
        self.assertEqual(artifacts, {})


if __name__ == "__main__":
    unittest.main()