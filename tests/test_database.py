"""
Tests for ProgramDatabase in openevolve.database
"""

import unittest
from openevolve.config import Config
from openevolve.database import Program, ProgramDatabase


class TestProgramDatabase(unittest.TestCase):
    """Tests for program database"""

    def setUp(self):
        """Set up test database"""
        config = Config()
        config.database.in_memory = True
        self.db = ProgramDatabase(config.database)

    def test_add_and_get(self):
        """Test adding and retrieving a program"""
        program = Program(
            id="test1",
            code="def test(): pass",
            language="python",
            metrics={"score": 0.5},
        )

        self.db.add(program)

        retrieved = self.db.get("test1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.id, "test1")
        self.assertEqual(retrieved.code, "def test(): pass")
        self.assertEqual(retrieved.metrics["score"], 0.5)

    def test_get_best_program(self):
        """Test getting the best program"""
        program1 = Program(
            id="test1",
            code="def test1(): pass",
            language="python",
            metrics={"score": 0.5},
        )

        program2 = Program(
            id="test2",
            code="def test2(): pass",
            language="python",
            metrics={"score": 0.7},
        )

        self.db.add(program1)
        self.db.add(program2)

        best = self.db.get_best_program()
        self.assertIsNotNone(best)
        self.assertEqual(best.id, "test2")

    def test_sample(self):
        """Test sampling from the database"""
        program1 = Program(
            id="test1",
            code="def test1(): pass",
            language="python",
            metrics={"score": 0.5},
        )

        program2 = Program(
            id="test2",
            code="def test2(): pass",
            language="python",
            metrics={"score": 0.7},
        )

        self.db.add(program1)
        self.db.add(program2)

        parent, inspirations = self.db.sample()

        self.assertIsNotNone(parent)
        self.assertIn(parent.id, ["test1", "test2"])

    def test_island_operations_basic(self):
        """Test basic island operations"""
        # Test with default islands (should be 5 by default)
        self.assertEqual(len(self.db.islands), 5)
        
        program = Program(
            id="island_test",
            code="def island_test(): pass",
            language="python",
            metrics={"score": 0.6},
        )
        
        self.db.add(program)
        
        # Should be in island 0
        self.assertIn("island_test", self.db.islands[0])
        self.assertEqual(program.metadata.get("island"), 0)

    def test_multi_island_setup(self):
        """Test database with multiple islands"""
        # Create new database with multiple islands
        config = Config()
        config.database.in_memory = True
        config.database.num_islands = 3
        multi_db = ProgramDatabase(config.database)
        
        self.assertEqual(len(multi_db.islands), 3)
        self.assertEqual(len(multi_db.island_best_programs), 3)
        
        # Add programs to specific islands
        for i in range(3):
            program = Program(
                id=f"test_island_{i}",
                code=f"def test_{i}(): pass",
                language="python",
                metrics={"score": 0.5 + i * 0.1},
            )
            multi_db.add(program, target_island=i)
            
            # Verify assignment
            self.assertIn(f"test_island_{i}", multi_db.islands[i])
            self.assertEqual(program.metadata.get("island"), i)

    def test_feature_coordinates_calculation(self):
        """Test MAP-Elites feature coordinate calculation"""
        program = Program(
            id="feature_test",
            code="def test(): pass",  # Short code
            language="python",
            metrics={"score": 0.8},
        )
        
        coords = self.db._calculate_feature_coords(program)
        
        # Should return list of coordinates
        self.assertIsInstance(coords, list)
        self.assertEqual(len(coords), len(self.db.config.feature_dimensions))
        
        # All coordinates should be within valid range
        for coord in coords:
            self.assertGreaterEqual(coord, 0)
            self.assertLess(coord, self.db.feature_bins)

    def test_feature_map_operations(self):
        """Test feature map operations for MAP-Elites"""
        program1 = Program(
            id="map_test1",
            code="def short(): pass",  # Similar complexity
            language="python",
            metrics={"score": 0.5},
        )
        
        program2 = Program(
            id="map_test2", 
            code="def also_short(): pass",  # Similar complexity
            language="python",
            metrics={"score": 0.8},  # Better score
        )
        
        self.db.add(program1)
        self.db.add(program2)
        
        # Both programs might land in same cell due to similar features
        # The better program should be kept in the feature map
        feature_coords1 = self.db._calculate_feature_coords(program1)
        feature_coords2 = self.db._calculate_feature_coords(program2)
        
        key1 = self.db._feature_coords_to_key(feature_coords1)
        key2 = self.db._feature_coords_to_key(feature_coords2)
        
        if key1 == key2:  # Same cell
            # Better program should be in feature map
            self.assertEqual(self.db.feature_map[key1], "map_test2")
        else:  # Different cells
            # Both should be in feature map
            self.assertEqual(self.db.feature_map[key1], "map_test1")
            self.assertEqual(self.db.feature_map[key2], "map_test2")

    def test_get_top_programs_with_metrics(self):
        """Test get_top_programs with specific metrics"""
        program1 = Program(
            id="metric_test1",
            code="def test1(): pass",
            language="python",
            metrics={"accuracy": 0.9, "speed": 0.3},
        )
        
        program2 = Program(
            id="metric_test2",
            code="def test2(): pass", 
            language="python",
            metrics={"accuracy": 0.7, "speed": 0.8},
        )
        
        self.db.add(program1)
        self.db.add(program2)
        
        # Test sorting by specific metric
        top_by_accuracy = self.db.get_top_programs(n=2, metric="accuracy")
        self.assertEqual(top_by_accuracy[0].id, "metric_test1")  # Higher accuracy
        
        top_by_speed = self.db.get_top_programs(n=2, metric="speed")
        self.assertEqual(top_by_speed[0].id, "metric_test2")  # Higher speed

    def test_archive_operations(self):
        """Test archive functionality"""
        # Add programs that should go into archive
        for i in range(5):
            program = Program(
                id=f"archive_test_{i}",
                code=f"def test_{i}(): return {i}",
                language="python",
                metrics={"score": i * 0.1},
            )
            self.db.add(program)
        
        # Archive should contain program IDs
        self.assertGreater(len(self.db.archive), 0)
        self.assertLessEqual(len(self.db.archive), self.db.config.archive_size)
        
        # Archive should contain program IDs that exist
        for program_id in self.db.archive:
            self.assertIn(program_id, self.db.programs)

    def test_best_program_tracking(self):
        """Test absolute best program tracking"""
        program1 = Program(
            id="best_test1",
            code="def test1(): pass",
            language="python",
            metrics={"combined_score": 0.6},
        )
        
        program2 = Program(
            id="best_test2",
            code="def test2(): pass",
            language="python", 
            metrics={"combined_score": 0.9},
        )
        
        self.db.add(program1)
        self.assertEqual(self.db.best_program_id, "best_test1")
        
        self.db.add(program2)
        self.assertEqual(self.db.best_program_id, "best_test2")  # Should update to better program

    def test_population_limit_enforcement(self):
        """Test population size limit enforcement"""
        # Set small population limit
        original_limit = self.db.config.population_size
        self.db.config.population_size = 3
        
        # Add more programs than limit
        for i in range(5):
            program = Program(
                id=f"limit_test_{i}",
                code=f"def test_{i}(): pass",
                language="python",
                metrics={"score": i * 0.1},
            )
            self.db.add(program)
        
        # Population should be at or below limit
        self.assertLessEqual(len(self.db.programs), 3)
        
        # Restore original limit
        self.db.config.population_size = original_limit

    def test_calculate_complexity_bin_adaptive(self):
        """Test adaptive complexity binning with multiple programs"""
        # Add programs with different complexities
        programs = [
            Program(id="short", code="x=1", metrics={"score": 0.5}),
            Program(id="medium", code="def func():\n    return x*2\n    pass", metrics={"score": 0.5}),
            Program(id="long", code="def complex_function():\n    result = []\n    for i in range(100):\n        result.append(i*2)\n    return result", metrics={"score": 0.5}),
        ]
        
        for program in programs:
            self.db.add(program)
        
        # Test binning for different complexity values
        short_bin = self.db._calculate_complexity_bin(len("x=1"))
        medium_bin = self.db._calculate_complexity_bin(len("def func():\n    return x*2\n    pass"))
        long_bin = self.db._calculate_complexity_bin(len("def complex_function():\n    result = []\n    for i in range(100):\n        result.append(i*2)\n    return result"))
        
        # Bins should be different and within valid range
        self.assertNotEqual(short_bin, long_bin)
        self.assertGreaterEqual(short_bin, 0)
        self.assertLess(short_bin, self.db.feature_bins)
        self.assertGreaterEqual(long_bin, 0)
        self.assertLess(long_bin, self.db.feature_bins)

    def test_calculate_complexity_bin_cold_start(self):
        """Test complexity binning during cold start (< 2 programs)"""
        # Empty database - should use fixed range
        bin_idx = self.db._calculate_complexity_bin(500)
        
        self.assertGreaterEqual(bin_idx, 0)
        self.assertLess(bin_idx, self.db.feature_bins)
        
        # Add one program - still cold start
        program = Program(id="single", code="x=1", metrics={"score": 0.5})
        self.db.add(program)
        
        bin_idx = self.db._calculate_complexity_bin(500)
        self.assertGreaterEqual(bin_idx, 0)
        self.assertLess(bin_idx, self.db.feature_bins)

    def test_calculate_diversity_bin_adaptive(self):
        """Test adaptive diversity binning with multiple programs"""
        # Add programs with different code structures for diversity testing
        programs = [
            Program(id="simple", code="x = 1", metrics={"score": 0.5}),
            Program(id="function", code="def add(a, b):\n    return a + b", metrics={"score": 0.5}),
            Program(id="loop", code="for i in range(10):\n    print(i)\n    x += i", metrics={"score": 0.5}),
            Program(id="complex", code="class MyClass:\n    def __init__(self):\n        self.data = []\n    def process(self, items):\n        return [x*2 for x in items]", metrics={"score": 0.5}),
        ]
        
        for program in programs:
            self.db.add(program)
        
        # Test binning for different diversity values
        # Use fast diversity to calculate test values
        simple_prog = programs[0]
        complex_prog = programs[3]
        
        # Calculate diversity for simple vs complex programs
        simple_diversity = self.db._fast_code_diversity(simple_prog.code, complex_prog.code)
        
        # Test the binning
        bin_idx = self.db._calculate_diversity_bin(simple_diversity)
        
        # Should be within valid range
        self.assertGreaterEqual(bin_idx, 0)
        self.assertLess(bin_idx, self.db.feature_bins)

    def test_calculate_diversity_bin_cold_start(self):
        """Test diversity binning during cold start (< 2 programs)"""
        # Empty database - should use fixed range
        bin_idx = self.db._calculate_diversity_bin(500.0)
        
        self.assertGreaterEqual(bin_idx, 0)
        self.assertLess(bin_idx, self.db.feature_bins)
        
        # Add one program - still cold start
        program = Program(id="single", code="x=1", metrics={"score": 0.5})
        self.db.add(program)
        
        bin_idx = self.db._calculate_diversity_bin(500.0)
        self.assertGreaterEqual(bin_idx, 0)
        self.assertLess(bin_idx, self.db.feature_bins)

    def test_calculate_diversity_bin_identical_programs(self):
        """Test diversity binning when all programs have identical diversity"""
        # Add multiple identical programs
        for i in range(3):
            program = Program(
                id=f"identical_{i}",
                code="x = 1",  # Same code
                metrics={"score": 0.5}
            )
            self.db.add(program)
        
        # Test binning - should handle zero range gracefully
        bin_idx = self.db._calculate_diversity_bin(0.0)
        
        self.assertGreaterEqual(bin_idx, 0)
        self.assertLess(bin_idx, self.db.feature_bins)

    def test_fast_code_diversity_function(self):
        """Test the _fast_code_diversity function"""
        # Test identical code
        code1 = "def test(): pass"
        code2 = "def test(): pass"
        diversity = self.db._fast_code_diversity(code1, code2)
        self.assertEqual(diversity, 0.0)
        
        # Test different code
        code1 = "x = 1"
        code2 = "def complex_function():\n    return [i*2 for i in range(100)]"
        diversity = self.db._fast_code_diversity(code1, code2)
        self.assertGreater(diversity, 0.0)
        
        # Test length difference
        short_code = "x = 1"
        long_code = "x = 1" + "a" * 100
        diversity = self.db._fast_code_diversity(short_code, long_code)
        self.assertGreater(diversity, 0.0)

    def test_diversity_feature_integration(self):
        """Test diversity feature calculation in feature coordinates"""
        # Add programs with different structures
        programs = [
            Program(id="prog1", code="x = 1", metrics={"score": 0.5}),
            Program(id="prog2", code="def func():\n    return 2", metrics={"score": 0.5}),
            Program(id="prog3", code="for i in range(5):\n    print(i)", metrics={"score": 0.5}),
        ]
        
        for program in programs:
            self.db.add(program)
        
        # Create a test program with diversity feature enabled
        test_config = self.db.config
        test_config.feature_dimensions = ["score", "complexity", "diversity"]
        
        test_program = Program(id="test", code="def test(): return 42", metrics={"score": 0.7})
        
        # Calculate feature coordinates - should include diversity dimension
        coords = self.db._calculate_feature_coords(test_program)
        
        # Should have 3 coordinates for score, complexity, and diversity
        self.assertEqual(len(coords), 3)
        
        # All coordinates should be within valid range
        for coord in coords:
            self.assertGreaterEqual(coord, 0)
            self.assertLess(coord, self.db.feature_bins)


if __name__ == "__main__":
    unittest.main()
