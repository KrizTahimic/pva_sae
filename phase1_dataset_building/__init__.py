"""
Phase 1: Dataset Building for the PVA-SAE project.

This package contains all components for building datasets by generating
Python solutions for MBPP problems and classifying them as correct or incorrect.

Main Components:
    - DatasetManager: MBPP dataset loading, management, and prompt generation
    - DatasetBuilder: Core dataset building logic
    - SolutionEvaluator: Code evaluation against test cases
"""

# Import dataset management components
from phase1_dataset_building.dataset_manager import (
    CodeTestResult,
    CodeGenerationResult,
    DatasetManager
)

# Import solution evaluation components
from phase1_dataset_building.solution_evaluator import (
    SolutionEvaluator,
    SafeSolutionEvaluator
)

# Import dataset building components
from phase1_dataset_building.dataset_builder import (
    CheckpointData,
    CheckpointManager,
    ProgressTracker,
    ResourceMonitor,
    DatasetBuilder,
    RobustDatasetBuilder
)

# Import main orchestrators
from phase1_dataset_building.mbpp_tester import (
    MBPPTester,
    DatasetBuildingOrchestrator,
    ProductionDatasetBuilder
)

__all__ = [
    # Dataset management - core data structures
    'CodeTestResult',
    'CodeGenerationResult', 
    'DatasetManager',
    
    # Solution evaluation - testing generated code
    'SolutionEvaluator',
    'SafeSolutionEvaluator',
    
    # Dataset building - core processing logic
    'CheckpointData',
    'CheckpointManager',
    'ProgressTracker', 
    'ResourceMonitor',
    'DatasetBuilder',
    'RobustDatasetBuilder',
    
    # Main orchestrators - high-level workflow coordination
    'MBPPTester',
    'DatasetBuildingOrchestrator',
    'ProductionDatasetBuilder'
]