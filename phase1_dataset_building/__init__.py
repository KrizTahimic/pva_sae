"""
Phase 1: Dataset Building for the PVA-SAE project.

This package contains all components for building datasets by generating
Python solutions for MBPP problems and classifying them as correct or incorrect.
"""

# Import dataset management components
from phase1_dataset_building.dataset_manager import (
    CodeTestResult,
    CodeGenerationResult,
    PromptTemplateBuilder,
    DatasetManager,
    PromptAwareDatasetManager
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
    # Dataset management
    'CodeTestResult',
    'CodeGenerationResult',
    'PromptTemplateBuilder',
    'DatasetManager',
    'PromptAwareDatasetManager',
    
    # Solution evaluation
    'SolutionEvaluator',
    'SafeSolutionEvaluator',
    
    # Dataset building
    'CheckpointData',
    'CheckpointManager',
    'ProgressTracker',
    'ResourceMonitor',
    'DatasetBuilder',
    'RobustDatasetBuilder',
    
    # Main orchestrators
    'MBPPTester',
    'DatasetBuildingOrchestrator',
    'ProductionDatasetBuilder'
]