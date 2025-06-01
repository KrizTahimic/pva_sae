"""
Logging utilities for the PVA-SAE project.

This module provides centralized logging configuration and utilities
for consistent logging across all project phases.
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
import json


class LoggingManager:
    """Manages logging configuration and setup for the project"""
    
    def __init__(self, 
                 log_dir: str = "mbpp_logs",
                 log_level: str = "INFO",
                 log_to_file: bool = True,
                 log_to_console: bool = True):
        """
        Initialize logging manager
        
        Args:
            log_dir: Directory for log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            log_to_file: Whether to log to file
            log_to_console: Whether to log to console
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper())
        self.log_to_file = log_to_file
        self.log_to_console = log_to_console
        self.log_file = None
        self.logger = None
        
    def setup_logging(self, logger_name: str = "pva_sae") -> logging.Logger:
        """
        Setup logging configuration
        
        Args:
            logger_name: Name for the logger
            
        Returns:
            logging.Logger: Configured logger instance
        """
        # Create logger
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.log_level)
        
        # Remove existing handlers
        self.logger.handlers = []
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Setup file handler
        if self.log_to_file:
            os.makedirs(self.log_dir, exist_ok=True)
            from common.utils import get_readable_timestamp
            timestamp = get_readable_timestamp()
            self.log_file = os.path.join(self.log_dir, f"pva_sae_{logger_name}_{timestamp}.log")
            
            file_handler = logging.FileHandler(self.log_file)
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(detailed_formatter)
            self.logger.addHandler(file_handler)
        
        # Setup console handler
        if self.log_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(simple_formatter)
            self.logger.addHandler(console_handler)
        
        # Log initialization
        self.logger.info(f"Logging initialized for {logger_name}")
        if self.log_file:
            self.logger.info(f"Log file: {self.log_file}")
        
        return self.logger
    
    def log_experiment_info(self, experiment_config: Dict[str, Any]):
        """
        Log experiment configuration and metadata
        
        Args:
            experiment_config: Dictionary with experiment configuration
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.info("="*60)
        self.logger.info("EXPERIMENT CONFIGURATION")
        self.logger.info("="*60)
        
        for key, value in experiment_config.items():
            if isinstance(value, dict):
                self.logger.info(f"{key}:")
                for sub_key, sub_value in value.items():
                    self.logger.info(f"  {sub_key}: {sub_value}")
            else:
                self.logger.info(f"{key}: {value}")
        
        self.logger.info("="*60)
    
    def log_phase_start(self, phase_name: str, total_items: Optional[int] = None):
        """
        Log the start of a processing phase
        
        Args:
            phase_name: Name of the phase
            total_items: Total number of items to process
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.info("")
        self.logger.info("*"*60)
        self.logger.info(f"STARTING PHASE: {phase_name.upper()}")
        if total_items:
            self.logger.info(f"Total items to process: {total_items}")
        self.logger.info("*"*60)
    
    def log_phase_end(self, phase_name: str, duration: float, success_count: int = 0, 
                      error_count: int = 0):
        """
        Log the end of a processing phase
        
        Args:
            phase_name: Name of the phase
            duration: Duration in seconds
            success_count: Number of successful items
            error_count: Number of failed items
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.info("")
        self.logger.info("*"*60)
        self.logger.info(f"COMPLETED PHASE: {phase_name.upper()}")
        self.logger.info(f"Duration: {duration:.2f} seconds")
        if success_count > 0 or error_count > 0:
            self.logger.info(f"Success: {success_count}, Errors: {error_count}")
            success_rate = success_count / (success_count + error_count) * 100
            self.logger.info(f"Success rate: {success_rate:.1f}%")
        self.logger.info("*"*60)
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """
        Log an error with additional context
        
        Args:
            error: The exception that occurred
            context: Dictionary with contextual information
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.error(f"Error occurred: {type(error).__name__}: {str(error)}")
        self.logger.error("Context:")
        for key, value in context.items():
            self.logger.error(f"  {key}: {value}")
        
        import traceback
        self.logger.error("Traceback:")
        self.logger.error(traceback.format_exc())
    
    def log_checkpoint(self, checkpoint_data: Dict[str, Any], checkpoint_file: str):
        """
        Log checkpoint save event
        
        Args:
            checkpoint_data: Data being checkpointed
            checkpoint_file: Path to checkpoint file
        """
        if not self.logger:
            self.setup_logging()
        
        self.logger.info(f"Checkpoint saved: {checkpoint_file}")
        self.logger.info(f"Checkpoint data: {len(checkpoint_data.get('processed_indices', []))} records processed")
    
    def get_log_file_path(self) -> Optional[str]:
        """Get the current log file path"""
        return self.log_file


class ExperimentLogger:
    """Specialized logger for experiment tracking"""
    
    def __init__(self, experiment_name: str, output_dir: str = "experiments"):
        """
        Initialize experiment logger
        
        Args:
            experiment_name: Name of the experiment
            output_dir: Directory for experiment outputs
        """
        self.experiment_name = experiment_name
        self.output_dir = output_dir
        self.start_time = datetime.now()
        self.metrics = {}
        self.events = []
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create experiment file
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.experiment_file = os.path.join(
            output_dir, 
            f"{experiment_name}_{timestamp}.json"
        )
    
    def log_metric(self, metric_name: str, value: Any, step: Optional[int] = None):
        """
        Log a metric value
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            step: Optional step/iteration number
        """
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        
        self.metrics[metric_name].append({
            "value": value,
            "step": step,
            "timestamp": datetime.now().isoformat()
        })
    
    def log_event(self, event_type: str, description: str, data: Optional[Dict] = None):
        """
        Log an event
        
        Args:
            event_type: Type of event
            description: Event description
            data: Optional additional data
        """
        self.events.append({
            "type": event_type,
            "description": description,
            "data": data,
            "timestamp": datetime.now().isoformat()
        })
    
    def save(self):
        """Save experiment data to file"""
        experiment_data = {
            "experiment_name": self.experiment_name,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "metrics": self.metrics,
            "events": self.events
        }
        
        with open(self.experiment_file, 'w') as f:
            json.dump(experiment_data, f, indent=2)
        
        print(f"Experiment data saved to: {self.experiment_file}")