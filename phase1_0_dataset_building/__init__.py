"""
Phase 1: Dataset Building for the PVA-SAE project.

This package contains all components for building datasets by generating
Python solutions for MBPP problems and classifying them as correct or incorrect.

Main Components:
    - DatasetManager: MBPP dataset loading, management, and prompt generation
    - DatasetBuilder: Core dataset building logic (simplified, no inheritance)
    - SolutionEvaluator: Code evaluation against test cases
    - Phase1Orchestrator: Single coordinator for the entire workflow
"""

# Import dataset management components
from phase1_0_dataset_building.dataset_manager import (
    CodeTestResult,
    CodeGenerationResult,
    DatasetManager
)

# Import solution evaluation components
from phase1_0_dataset_building.solution_evaluator import (
    SolutionEvaluator,
    SafeSolutionEvaluator
)

# Import dataset building components
from phase1_0_dataset_building.dataset_builder import DatasetBuilder

# Import orchestrator
from phase1_0_dataset_building.orchestrator import Phase1Orchestrator

# Import utility modules (not classes, just modules)
from phase1_0_dataset_building import checkpoint_manager
from phase1_0_dataset_building import resource_monitor

__all__ = [
    # Dataset management - core data structures
    'CodeTestResult',
    'CodeGenerationResult', 
    'DatasetManager',
    
    # Solution evaluation - testing generated code
    'SolutionEvaluator',
    'SafeSolutionEvaluator',
    
    # Dataset building - core processing logic
    'DatasetBuilder',
    
    # Main orchestrator - single coordinator
    'Phase1Orchestrator',
    
    # Utility modules
    'checkpoint_manager',
    'resource_monitor'
]