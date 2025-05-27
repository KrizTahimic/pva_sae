"""
Phase 1: Dataset Building for the PVA-SAE project.

This package contains all components for building datasets by generating
Python solutions for MBPP problems and classifying them as correct or incorrect.
"""

# Import dataset management components
from phase1_dataset_building.dataset_manager import (
    TestResult,
    GenerationResult,
    PromptTemplateBuilder,
    DatasetManager,
    EnhancedDatasetManager
)

# Import test execution components
from phase1_dataset_building.test_executor import (
    TestExecutor,
    EnhancedTestExecutor
)

# Import dataset building components
from phase1_dataset_building.dataset_builder import (
    CheckpointData,
    CheckpointManager,
    ProgressTracker,
    ResourceMonitor,
    DatasetBuilder,
    HardenedDatasetBuilder
)

# Import main orchestrators
from phase1_dataset_building.mbpp_tester import (
    MBPPTester,
    EnhancedMBPPTester,
    ProductionMBPPTester
)

__all__ = [
    # Dataset management
    'TestResult',
    'GenerationResult',
    'PromptTemplateBuilder',
    'DatasetManager',
    'EnhancedDatasetManager',
    
    # Test execution
    'TestExecutor',
    'EnhancedTestExecutor',
    
    # Dataset building
    'CheckpointData',
    'CheckpointManager',
    'ProgressTracker',
    'ResourceMonitor',
    'DatasetBuilder',
    'HardenedDatasetBuilder',
    
    # Main orchestrators
    'MBPPTester',
    'EnhancedMBPPTester',
    'ProductionMBPPTester'
]