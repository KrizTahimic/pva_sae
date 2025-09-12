"""
Statistical Significance Tester for Phase 4.14.

Validates that targeted PVA steering significantly outperforms zero-discrimination
steering using binomial tests, proving causal effects of PVA features.
"""

import json
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from scipy.stats import binomtest
import numpy as np

from common.logging import get_logger
from common.utils import (
    discover_latest_phase_output,
    ensure_directory_exists
)
from common_simplified.helpers import load_json, save_json
from common.config import Config

logger = get_logger("phase4_14.significance_tester")


class SignificanceTester:
    """Test statistical significance between targeted and zero-discrimination steering."""
    
    def __init__(self, config: Config):
        """Initialize with configuration."""
        self.config = config
        
        # Output directory
        self.output_dir = Path(config.phase4_14_output_dir)
        ensure_directory_exists(self.output_dir)
        
        # Significance level
        self.alpha = getattr(config, 'phase4_14_significance_level', 0.05)
        
        logger.info(f"SignificanceTester initialized")
        logger.info(f"Significance level: {self.alpha}")
        
    def load_steering_results(self) -> Tuple[Dict, Dict]:
        """Load Phase 4.8 targeted and Phase 4.12 zero-discrimination results."""
        # Load Phase 4.8 targeted steering results
        logger.info("Loading Phase 4.8 targeted steering results...")
        phase4_8_output = discover_latest_phase_output("4.8")
        if not phase4_8_output:
            raise FileNotFoundError("Phase 4.8 output not found. Run Phase 4.8 first.")
        
        targeted_file = Path(phase4_8_output).parent / "steering_effect_analysis.json"
        if not targeted_file.exists():
            raise FileNotFoundError(f"Targeted steering results not found at {targeted_file}")
        
        targeted_results = load_json(targeted_file)
        logger.info(f"Loaded targeted steering results")
        
        # Load Phase 4.12 zero-discrimination steering results
        logger.info("Loading Phase 4.12 zero-discrimination steering results...")
        phase4_12_output = discover_latest_phase_output("4.12")
        if not phase4_12_output:
            raise FileNotFoundError("Phase 4.12 output not found. Run Phase 4.12 first.")
        
        zero_disc_file = Path(phase4_12_output).parent / "zero_disc_steering_results.json"
        if not zero_disc_file.exists():
            # Try legacy filename
            zero_disc_file = Path(phase4_12_output).parent / "random_steering_results.json"
            if not zero_disc_file.exists():
                raise FileNotFoundError(f"Zero-discrimination results not found at {zero_disc_file}")
        
        zero_disc_results = load_json(zero_disc_file)
        logger.info(f"Loaded zero-discrimination steering results")
        
        return targeted_results, zero_disc_results
        
    def perform_binomial_test(self, n_successes: int, n_trials: int, 
                            baseline_rate: float, alternative: str = 'greater') -> Dict:
        """Perform binomial test comparing observed vs expected rate."""
        if n_trials == 0:
            logger.warning("No trials available for binomial test")
            return {
                'n_successes': 0,
                'n_trials': 0,
                'observed_rate': 0.0,
                'expected_rate': baseline_rate,
                'p_value': 1.0,
                'significant': False,
                'alternative': alternative
            }
        
        # Perform binomial test
        result = binomtest(n_successes, n_trials, p=baseline_rate, alternative=alternative)
        
        observed_rate = n_successes / n_trials
        
        return {
            'n_successes': n_successes,
            'n_trials': n_trials,
            'observed_rate': observed_rate,
            'expected_rate': baseline_rate,
            'p_value': result.pvalue,
            'significant': result.pvalue < self.alpha,
            'alternative': alternative,
            'confidence_interval': result.proportion_ci(confidence_level=1-self.alpha)
        }
        
    def analyze_correction_significance(self, targeted: Dict, zero_disc: Dict) -> Dict:
        """Analyze significance of correction experiments (incorrect→correct)."""
        # Extract targeted correction results
        if 'correction' in targeted:
            targeted_data = targeted['correction']
            targeted_n_corrected = targeted_data.get('n_corrected', 0)
            targeted_n_total = targeted_data.get('n_total', 0)
            targeted_rate = targeted_data.get('rate', 0.0)
        else:
            # Fallback structure
            targeted_n_corrected = targeted.get('summary_metrics', {}).get('n_corrected', 0)
            targeted_n_total = len(targeted.get('correction_results', {}))
            targeted_rate = targeted_n_corrected / targeted_n_total if targeted_n_total > 0 else 0.0
        
        # Extract zero-disc correction results
        zero_disc_correction = zero_disc.get('correction_results', {})
        zero_disc_n_corrected = sum(1 for r in zero_disc_correction.values() 
                                   if r['steered_correct'] and not r['initial_correct'])
        zero_disc_n_total = len(zero_disc_correction)
        zero_disc_rate = zero_disc_n_corrected / zero_disc_n_total if zero_disc_n_total > 0 else 0.0
        
        logger.info(f"Correction rates - Targeted: {targeted_rate:.2%}, Zero-disc: {zero_disc_rate:.2%}")
        
        # Perform binomial test: Is targeted significantly better than zero-disc?
        test_result = self.perform_binomial_test(
            targeted_n_corrected,
            targeted_n_total,
            zero_disc_rate,
            alternative='greater'
        )
        
        return {
            'zero_discrimination': {
                'n_corrected': zero_disc_n_corrected,
                'n_total': zero_disc_n_total,
                'rate': zero_disc_rate
            },
            'targeted': {
                'n_corrected': targeted_n_corrected,
                'n_total': targeted_n_total,
                'rate': targeted_rate
            },
            'binomial_test': test_result,
            'interpretation': self._interpret_correction_result(test_result, targeted_rate, zero_disc_rate)
        }
        
    def analyze_corruption_significance(self, targeted: Dict, zero_disc: Dict) -> Dict:
        """Analyze significance of corruption experiments (correct→incorrect)."""
        # Extract targeted corruption results
        if 'corruption' in targeted:
            targeted_data = targeted['corruption']
            targeted_n_corrupted = targeted_data.get('n_corrupted', 0)
            targeted_n_total = targeted_data.get('n_total', 0)
            targeted_rate = targeted_data.get('rate', 0.0)
        else:
            # Fallback structure
            targeted_n_corrupted = targeted.get('summary_metrics', {}).get('n_corrupted', 0)
            targeted_n_total = len(targeted.get('corruption_results', {}))
            targeted_rate = targeted_n_corrupted / targeted_n_total if targeted_n_total > 0 else 0.0
        
        # Extract zero-disc corruption results
        zero_disc_corruption = zero_disc.get('corruption_results', {})
        zero_disc_n_corrupted = sum(1 for r in zero_disc_corruption.values() 
                                   if not r['steered_correct'] and r['initial_correct'])
        zero_disc_n_total = len(zero_disc_corruption)
        zero_disc_rate = zero_disc_n_corrupted / zero_disc_n_total if zero_disc_n_total > 0 else 0.0
        
        logger.info(f"Corruption rates - Targeted: {targeted_rate:.2%}, Zero-disc: {zero_disc_rate:.2%}")
        
        # Perform binomial test: Is targeted significantly better than zero-disc?
        test_result = self.perform_binomial_test(
            targeted_n_corrupted,
            targeted_n_total,
            zero_disc_rate,
            alternative='greater'
        )
        
        return {
            'zero_discrimination': {
                'n_corrupted': zero_disc_n_corrupted,
                'n_total': zero_disc_n_total,
                'rate': zero_disc_rate
            },
            'targeted': {
                'n_corrupted': targeted_n_corrupted,
                'n_total': targeted_n_total,
                'rate': targeted_rate
            },
            'binomial_test': test_result,
            'interpretation': self._interpret_corruption_result(test_result, targeted_rate, zero_disc_rate)
        }
        
    def _interpret_correction_result(self, test_result: Dict, targeted_rate: float, zero_disc_rate: float) -> str:
        """Generate interpretation for correction test results."""
        if test_result['significant']:
            effect_size = targeted_rate - zero_disc_rate
            return (f"Targeted steering significantly outperforms zero-discrimination baseline "
                   f"(p={test_result['p_value']:.2e}, effect size={effect_size:.2%}). "
                   f"This validates that PVA features have specific causal effects on program correctness.")
        else:
            return (f"No significant difference between targeted and zero-discrimination steering "
                   f"(p={test_result['p_value']:.3f}). Further investigation may be needed.")
            
    def _interpret_corruption_result(self, test_result: Dict, targeted_rate: float, zero_disc_rate: float) -> str:
        """Generate interpretation for corruption test results."""
        if test_result['significant']:
            effect_size = targeted_rate - zero_disc_rate
            return (f"Targeted incorrect-preferring features significantly corrupt more than baseline "
                   f"(p={test_result['p_value']:.2e}, effect size={effect_size:.2%}). "
                   f"This confirms the semantic specificity of PVA features.")
        else:
            return (f"No significant difference in corruption rates "
                   f"(p={test_result['p_value']:.3f}). May indicate weaker incorrect-preferring features.")
            
    def run(self) -> Dict:
        """Run statistical significance testing."""
        logger.info("="*60)
        logger.info("Starting Statistical Significance Testing")
        logger.info("="*60)
        
        # Import here to avoid circular imports
        from typing import Tuple
        
        # Load results from both phases
        targeted_results, zero_disc_results = self.load_steering_results()
        
        # Analyze correction significance
        logger.info("\n" + "="*40)
        logger.info("Analyzing CORRECTION significance")
        logger.info("="*40)
        correction_analysis = self.analyze_correction_significance(targeted_results, zero_disc_results)
        
        # Analyze corruption significance
        logger.info("\n" + "="*40)
        logger.info("Analyzing CORRUPTION significance")
        logger.info("="*40)
        corruption_analysis = self.analyze_corruption_significance(targeted_results, zero_disc_results)
        
        # Calculate overall conclusion
        both_significant = (correction_analysis['binomial_test']['significant'] and 
                          corruption_analysis['binomial_test']['significant'])
        
        if both_significant:
            conclusion = ("PVA features have specific causal effects on program correctness, "
                        "not explained by arbitrary activation modifications. "
                        "Both correction and corruption experiments show significant differences "
                        "between targeted and zero-discrimination steering.")
        elif correction_analysis['binomial_test']['significant']:
            conclusion = ("Targeted steering shows significant correction effects compared to baseline, "
                        "but corruption effects are not significantly different. "
                        "This suggests asymmetric steering capabilities.")
        elif corruption_analysis['binomial_test']['significant']:
            conclusion = ("Targeted steering shows significant corruption effects compared to baseline, "
                        "but correction effects are not significantly different. "
                        "This suggests stronger incorrect-preferring features.")
        else:
            conclusion = ("No significant differences found between targeted and zero-discrimination steering. "
                        "This may indicate that the selected features or coefficients need refinement.")
        
        # Prepare results
        results = {
            'metadata': {
                'phase': '4.14',
                'description': 'Statistical significance testing for steering validation',
                'test_type': 'binomial',
                'alternative': 'greater',
                'significance_level': self.alpha,
                'timestamp': datetime.now().isoformat()
            },
            'correction_significance': correction_analysis,
            'corruption_significance': corruption_analysis,
            'conclusion': conclusion,
            'summary': {
                'correction_p_value': correction_analysis['binomial_test']['p_value'],
                'corruption_p_value': corruption_analysis['binomial_test']['p_value'],
                'both_significant': both_significant,
                'targeted_correction_rate': correction_analysis['targeted']['rate'],
                'zero_disc_correction_rate': correction_analysis['zero_discrimination']['rate'],
                'targeted_corruption_rate': corruption_analysis['targeted']['rate'],
                'zero_disc_corruption_rate': corruption_analysis['zero_discrimination']['rate']
            }
        }
        
        # Save results
        output_file = self.output_dir / 'statistical_significance.json'
        save_json(results, output_file)
        logger.info(f"Saved results to: {output_file}")
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("STATISTICAL SIGNIFICANCE RESULTS")
        logger.info("="*60)
        logger.info(f"Correction test:")
        logger.info(f"  Targeted: {correction_analysis['targeted']['rate']:.2%}")
        logger.info(f"  Zero-disc: {correction_analysis['zero_discrimination']['rate']:.2%}")
        logger.info(f"  p-value: {correction_analysis['binomial_test']['p_value']:.2e}")
        logger.info(f"  Significant: {correction_analysis['binomial_test']['significant']}")
        logger.info("")
        logger.info(f"Corruption test:")
        logger.info(f"  Targeted: {corruption_analysis['targeted']['rate']:.2%}")
        logger.info(f"  Zero-disc: {corruption_analysis['zero_discrimination']['rate']:.2%}")
        logger.info(f"  p-value: {corruption_analysis['binomial_test']['p_value']:.2e}")
        logger.info(f"  Significant: {corruption_analysis['binomial_test']['significant']}")
        logger.info("")
        logger.info(f"Conclusion: {conclusion}")
        logger.info("="*60)
        
        return results