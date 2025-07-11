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


if __name__ == "__main__":
    unittest.main()
