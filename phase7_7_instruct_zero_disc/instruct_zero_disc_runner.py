"""
Instruction-tuned model zero-discrimination steering runner for Phase 7.7.

Thin wrapper around Phase 4.12's ZeroDiscSteeringGenerator that runs with
the instruction-tuned model variant. Provides a proper zero-disc control
condition for comparing against Phase 7.6 targeted instruct steering.
"""

from common.config import Config
from common.logging import get_logger

logger = get_logger("phase7_7.instruct_zero_disc_runner")


class InstructZeroDiscRunner:
    """Run zero-disc steering on the instruction-tuned model."""

    def __init__(self, config: Config, gpu_id: int = 0, n_gpus: int = 1):
        """Initialize by creating a ZeroDiscSteeringGenerator with instruct model override.

        Args:
            config: Configuration object
            gpu_id: GPU index for parallel execution (0-indexed)
            n_gpus: Total number of GPUs (1 = sequential)
        """
        from phase4_12_zero_disc_steering.zero_disc_steering_generator import ZeroDiscSteeringGenerator

        instruct_model = config.phase7_7_model_name
        logger.info(f"Phase 7.7: Zero-disc steering with instruct model: {instruct_model}")
        logger.info(f"Features from base model config (Phase 4.10), model override for generation only")

        self.generator = ZeroDiscSteeringGenerator(
            config=config,
            gpu_id=gpu_id,
            n_gpus=n_gpus,
            model_name_override=instruct_model,
            output_phase='7.7'
        )

    def run(self) -> dict:
        """Run zero-disc steering on the instruct model."""
        return self.generator.run()
