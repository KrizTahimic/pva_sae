"""
Statistical Significance Tester for Phase 4.14.

Performs triangulation analysis comparing baseline (no steering), zero-discrimination
steering, and targeted PVA steering to validate causal effects of PVA features.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from datetime import datetime
from scipy.stats import binomtest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from common.logging import get_logger
from common.utils import (
    discover_latest_phase_output,
    ensure_directory_exists
)
from common_simplified.helpers import load_json, save_json
from common.config import Config

logger = get_logger("phase4_14.significance_tester")


class SignificanceTester:
    """Test statistical significance using triangulation of three conditions."""
    
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
        
    def load_all_results(self) -> Tuple[pd.DataFrame, Dict, Dict]:
        """Load baseline, targeted steering, and zero-discrimination results."""
        # Load Phase 3.5 baseline data (no steering)
        logger.info("Loading Phase 3.5 baseline data (no steering)...")
        phase3_5_output = discover_latest_phase_output("3.5")
        if not phase3_5_output:
            raise FileNotFoundError("Phase 3.5 output not found. Run Phase 3.5 first.")
        
        baseline_file = Path(phase3_5_output).parent / "dataset_temp_0_0.parquet"
        if not baseline_file.exists():
            raise FileNotFoundError(f"Baseline data not found at {baseline_file}")
        
        baseline_data = pd.read_parquet(baseline_file)
        logger.info(f"Loaded {len(baseline_data)} baseline problems")
        
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
        
        return baseline_data, targeted_results, zero_disc_results
        
    def extract_baseline_metrics(self, baseline_data: pd.DataFrame) -> Dict:
        """Extract correction and corruption rates from baseline data."""
        # Split by correctness
        baseline_correct = baseline_data[baseline_data['test_passed'] == True]
        baseline_incorrect = baseline_data[baseline_data['test_passed'] == False]
        
        return {
            'n_correct': len(baseline_correct),
            'n_incorrect': len(baseline_incorrect),
            'n_total': len(baseline_data),
            'correct_rate': len(baseline_correct) / len(baseline_data) if len(baseline_data) > 0 else 0.0,
            'incorrect_rate': len(baseline_incorrect) / len(baseline_data) if len(baseline_data) > 0 else 0.0
        }
        
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
                'alternative': alternative,
                'effect_size': 0.0,
                'confidence_interval': (0.0, 0.0)
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
            'confidence_interval': result.proportion_ci(confidence_level=1-self.alpha),
            'effect_size': observed_rate - baseline_rate
        }
        
    def perform_correction_triangulation(self, baseline_data: pd.DataFrame, 
                                       targeted: Dict, zero_disc: Dict) -> Dict:
        """Perform triangulation for correction experiments (incorrect→correct)."""
        # Get baseline incorrect problems
        baseline_incorrect = baseline_data[baseline_data['test_passed'] == False]
        baseline_n_incorrect = len(baseline_incorrect)
        # Since baseline has no steering, correction rate is 0
        baseline_correction_rate = 0.0
        
        # Extract targeted correction results from Phase 4.8
        # Phase 4.8 structure: detailed_results -> correction (list)
        targeted_correction_list = targeted.get('detailed_results', {}).get('correction', [])
        if targeted_correction_list:
            # Count actual corrections (incorrect -> correct transitions)
            targeted_n_corrected = sum(1 for r in targeted_correction_list 
                                      if r.get('steered_passed') and not r.get('test_passed'))
            targeted_n_total = len(targeted_correction_list)
        else:
            # Fallback for empty results
            targeted_n_corrected = 0
            targeted_n_total = 0
        
        targeted_correction_rate = targeted_n_corrected / targeted_n_total if targeted_n_total > 0 else 0.0
        
        # Extract zero-disc correction results
        zero_disc_correction = zero_disc.get('correction_results', {})
        zero_disc_n_corrected = sum(1 for r in zero_disc_correction.values() 
                                   if r['steered_correct'] and not r['initial_correct'])
        zero_disc_n_total = len(zero_disc_correction)
        zero_disc_correction_rate = zero_disc_n_corrected / zero_disc_n_total if zero_disc_n_total > 0 else 0.0
        
        logger.info(f"Correction rates - Baseline: {baseline_correction_rate:.2%}, "
                   f"Zero-disc: {zero_disc_correction_rate:.2%}, "
                   f"Targeted: {targeted_correction_rate:.2%}")
        
        # Perform three pairwise comparisons
        comparisons = {}
        
        # 1. Baseline vs Targeted: Does targeted steering work at all?
        comparisons['baseline_vs_targeted'] = self.perform_binomial_test(
            targeted_n_corrected,
            targeted_n_total,
            baseline_correction_rate,
            alternative='greater'
        )
        
        # 2. Zero-disc vs Targeted: Is the effect specific to discriminative features?
        comparisons['zero_disc_vs_targeted'] = self.perform_binomial_test(
            targeted_n_corrected,
            targeted_n_total,
            zero_disc_correction_rate,
            alternative='greater'
        )
        
        # 3. Baseline vs Zero-disc: Do random features have minimal effect?
        comparisons['baseline_vs_zero_disc'] = self.perform_binomial_test(
            zero_disc_n_corrected,
            zero_disc_n_total,
            baseline_correction_rate,
            alternative='greater'
        )
        
        return {
            'rates': {
                'baseline': baseline_correction_rate,
                'zero_discrimination': zero_disc_correction_rate,
                'targeted': targeted_correction_rate
            },
            'counts': {
                'baseline': {'n_corrected': 0, 'n_total': baseline_n_incorrect},
                'zero_discrimination': {'n_corrected': zero_disc_n_corrected, 'n_total': zero_disc_n_total},
                'targeted': {'n_corrected': targeted_n_corrected, 'n_total': targeted_n_total}
            },
            'comparisons': comparisons
        }
        
    def perform_corruption_triangulation(self, baseline_data: pd.DataFrame, 
                                       targeted: Dict, zero_disc: Dict) -> Dict:
        """Perform triangulation for corruption experiments (correct→incorrect)."""
        # Get baseline correct problems
        baseline_correct = baseline_data[baseline_data['test_passed'] == True]
        baseline_n_correct = len(baseline_correct)
        # Since baseline has no steering, corruption rate is 0
        baseline_corruption_rate = 0.0
        
        # Extract targeted corruption results from Phase 4.8
        # Phase 4.8 structure: detailed_results -> corruption (list)
        targeted_corruption_list = targeted.get('detailed_results', {}).get('corruption', [])
        if targeted_corruption_list:
            # Count actual corruptions (correct -> incorrect transitions)
            targeted_n_corrupted = sum(1 for r in targeted_corruption_list 
                                      if not r.get('steered_passed') and r.get('test_passed'))
            targeted_n_total = len(targeted_corruption_list)
        else:
            # Fallback for empty results
            targeted_n_corrupted = 0
            targeted_n_total = 0
        
        targeted_corruption_rate = targeted_n_corrupted / targeted_n_total if targeted_n_total > 0 else 0.0
        
        # Extract zero-disc corruption results
        zero_disc_corruption = zero_disc.get('corruption_results', {})
        zero_disc_n_corrupted = sum(1 for r in zero_disc_corruption.values() 
                                   if not r['steered_correct'] and r['initial_correct'])
        zero_disc_n_total = len(zero_disc_corruption)
        zero_disc_corruption_rate = zero_disc_n_corrupted / zero_disc_n_total if zero_disc_n_total > 0 else 0.0
        
        logger.info(f"Corruption rates - Baseline: {baseline_corruption_rate:.2%}, "
                   f"Zero-disc: {zero_disc_corruption_rate:.2%}, "
                   f"Targeted: {targeted_corruption_rate:.2%}")
        
        # Perform three pairwise comparisons
        comparisons = {}
        
        # 1. Baseline vs Targeted: Does targeted steering corrupt?
        comparisons['baseline_vs_targeted'] = self.perform_binomial_test(
            targeted_n_corrupted,
            targeted_n_total,
            baseline_corruption_rate,
            alternative='greater'
        )
        
        # 2. Zero-disc vs Targeted: Is corruption specific to discriminative features?
        comparisons['zero_disc_vs_targeted'] = self.perform_binomial_test(
            targeted_n_corrupted,
            targeted_n_total,
            zero_disc_corruption_rate,
            alternative='greater'
        )
        
        # 3. Baseline vs Zero-disc: Do random features corrupt minimally?
        comparisons['baseline_vs_zero_disc'] = self.perform_binomial_test(
            zero_disc_n_corrupted,
            zero_disc_n_total,
            baseline_corruption_rate,
            alternative='greater'
        )
        
        return {
            'rates': {
                'baseline': baseline_corruption_rate,
                'zero_discrimination': zero_disc_corruption_rate,
                'targeted': targeted_corruption_rate
            },
            'counts': {
                'baseline': {'n_corrupted': 0, 'n_total': baseline_n_correct},
                'zero_discrimination': {'n_corrupted': zero_disc_n_corrupted, 'n_total': zero_disc_n_total},
                'targeted': {'n_corrupted': targeted_n_corrupted, 'n_total': targeted_n_total}
            },
            'comparisons': comparisons
        }
        
    def interpret_triangulation(self, correction_tri: Dict, corruption_tri: Dict) -> Dict:
        """Generate comprehensive interpretation of triangulation results."""
        correction_comps = correction_tri['comparisons']
        corruption_comps = corruption_tri['comparisons']
        
        # Check validity conditions
        validity_checks = {
            'steering_works': (
                correction_comps['baseline_vs_targeted']['significant'] or
                corruption_comps['baseline_vs_targeted']['significant']
            ),
            'feature_specific': (
                correction_comps['zero_disc_vs_targeted']['significant'] or
                corruption_comps['zero_disc_vs_targeted']['significant']
            ),
            'controls_valid': (
                not correction_comps['baseline_vs_zero_disc']['significant'] or
                correction_comps['baseline_vs_zero_disc']['effect_size'] < 0.03
            ) and (
                not corruption_comps['baseline_vs_zero_disc']['significant'] or
                corruption_comps['baseline_vs_zero_disc']['effect_size'] < 0.03
            )
        }
        
        # Generate interpretation based on pattern of results
        if validity_checks['steering_works'] and validity_checks['feature_specific'] and validity_checks['controls_valid']:
            interpretation = (
                "Strong validation: Targeted PVA steering shows significant effects compared to baseline, "
                "these effects are specific to discriminative features (not present with random features), "
                "and the control comparison confirms random features have minimal impact. "
                "This triangulation robustly validates that PVA features have specific causal effects on program correctness."
            )
            validation_strength = "STRONG"
        elif validity_checks['steering_works'] and validity_checks['feature_specific']:
            interpretation = (
                "Moderate validation: Targeted steering works and is feature-specific, "
                "but the control comparison shows some unexpected effects from random features. "
                "The main hypothesis is supported but controls may need refinement."
            )
            validation_strength = "MODERATE"
        elif validity_checks['steering_works']:
            interpretation = (
                "Weak validation: Targeted steering shows effects compared to baseline, "
                "but these effects are not significantly different from random feature steering. "
                "This suggests the steering mechanism works but may not be specific to PVA features."
            )
            validation_strength = "WEAK"
        else:
            interpretation = (
                "No validation: Targeted steering does not show significant effects. "
                "This may indicate issues with the selected features, steering coefficients, "
                "or the underlying hypothesis. Further investigation is needed."
            )
            validation_strength = "NONE"
        
        return {
            'validity_checks': validity_checks,
            'validation_strength': validation_strength,
            'interpretation': interpretation,
            'detailed_findings': {
                'correction': self._interpret_correction_triangulation(correction_comps),
                'corruption': self._interpret_corruption_triangulation(corruption_comps)
            }
        }
        
    def _interpret_correction_triangulation(self, comparisons: Dict) -> str:
        """Generate detailed interpretation for correction triangulation."""
        findings = []
        
        if comparisons['baseline_vs_targeted']['significant']:
            effect = comparisons['baseline_vs_targeted']['effect_size']
            findings.append(f"Targeted steering significantly corrects errors ({effect:.1%} improvement, "
                          f"p={comparisons['baseline_vs_targeted']['p_value']:.2e})")
        else:
            findings.append("Targeted steering does not significantly correct errors")
        
        if comparisons['zero_disc_vs_targeted']['significant']:
            effect = comparisons['zero_disc_vs_targeted']['effect_size']
            findings.append(f"Targeted outperforms random features ({effect:.1%} advantage, "
                          f"p={comparisons['zero_disc_vs_targeted']['p_value']:.2e})")
        else:
            findings.append("Targeted and random features perform similarly")
        
        if comparisons['baseline_vs_zero_disc']['significant']:
            effect = comparisons['baseline_vs_zero_disc']['effect_size']
            if effect > 0.03:
                findings.append(f"WARNING: Random features show unexpected correction ({effect:.1%}, "
                              f"p={comparisons['baseline_vs_zero_disc']['p_value']:.2e})")
            else:
                findings.append(f"Random features show minimal correction ({effect:.1%})")
        else:
            findings.append("Random features have no significant correction effect (expected)")
        
        return " | ".join(findings)
        
    def _interpret_corruption_triangulation(self, comparisons: Dict) -> str:
        """Generate detailed interpretation for corruption triangulation."""
        findings = []
        
        if comparisons['baseline_vs_targeted']['significant']:
            effect = comparisons['baseline_vs_targeted']['effect_size']
            findings.append(f"Incorrect-preferring features significantly corrupt ({effect:.1%} corruption, "
                          f"p={comparisons['baseline_vs_targeted']['p_value']:.2e})")
        else:
            findings.append("Incorrect-preferring features do not significantly corrupt")
        
        if comparisons['zero_disc_vs_targeted']['significant']:
            effect = comparisons['zero_disc_vs_targeted']['effect_size']
            findings.append(f"Targeted corruption exceeds random ({effect:.1%} more, "
                          f"p={comparisons['zero_disc_vs_targeted']['p_value']:.2e})")
        else:
            findings.append("Targeted and random corruption rates similar")
        
        if comparisons['baseline_vs_zero_disc']['significant']:
            effect = comparisons['baseline_vs_zero_disc']['effect_size']
            if effect > 0.03:
                findings.append(f"WARNING: Random features corrupt unexpectedly ({effect:.1%}, "
                              f"p={comparisons['baseline_vs_zero_disc']['p_value']:.2e})")
            else:
                findings.append(f"Random features show minimal corruption ({effect:.1%})")
        else:
            findings.append("Random features have no significant corruption effect (expected)")
        
        return " | ".join(findings)
        
    def create_visualization(self, correction_tri: Dict, corruption_tri: Dict) -> None:
        """Create bar plots comparing all three conditions."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Correction plot
        conditions = ['Baseline\n(No Steering)', 'Zero-Disc\n(Random)', 'Targeted\n(PVA)']
        correction_rates = [
            correction_tri['rates']['baseline'] * 100,
            correction_tri['rates']['zero_discrimination'] * 100,
            correction_tri['rates']['targeted'] * 100
        ]
        
        bars1 = ax1.bar(conditions, correction_rates, color=['gray', 'orange', 'green'])
        ax1.set_ylabel('Correction Rate (%)', fontsize=12)
        ax1.set_title('Correction Experiments (Incorrect→Correct)', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, max(correction_rates) * 1.2)
        
        # Add significance indicators
        if correction_tri['comparisons']['baseline_vs_targeted']['significant']:
            ax1.plot([0, 2], [max(correction_rates)*1.1]*2, 'k-', linewidth=1)
            ax1.text(1, max(correction_rates)*1.12, '***', ha='center', fontsize=14)
        
        if correction_tri['comparisons']['zero_disc_vs_targeted']['significant']:
            ax1.plot([1, 2], [max(correction_rates)*1.05]*2, 'k-', linewidth=1)
            ax1.text(1.5, max(correction_rates)*1.07, '**', ha='center', fontsize=14)
        
        # Add value labels on bars
        for bar, rate in zip(bars1, correction_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # Corruption plot
        corruption_rates = [
            corruption_tri['rates']['baseline'] * 100,
            corruption_tri['rates']['zero_discrimination'] * 100,
            corruption_tri['rates']['targeted'] * 100
        ]
        
        bars2 = ax2.bar(conditions, corruption_rates, color=['gray', 'orange', 'red'])
        ax2.set_ylabel('Corruption Rate (%)', fontsize=12)
        ax2.set_title('Corruption Experiments (Correct→Incorrect)', fontsize=14, fontweight='bold')
        ax2.set_ylim(0, max(corruption_rates) * 1.2 if max(corruption_rates) > 0 else 5)
        
        # Add significance indicators
        if corruption_tri['comparisons']['baseline_vs_targeted']['significant']:
            ax2.plot([0, 2], [max(corruption_rates)*1.1]*2, 'k-', linewidth=1)
            ax2.text(1, max(corruption_rates)*1.12, '***', ha='center', fontsize=14)
        
        if corruption_tri['comparisons']['zero_disc_vs_targeted']['significant']:
            ax2.plot([1, 2], [max(corruption_rates)*1.05]*2, 'k-', linewidth=1)
            ax2.text(1.5, max(corruption_rates)*1.07, '**', ha='center', fontsize=14)
        
        # Add value labels on bars
        for bar, rate in zip(bars2, corruption_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.suptitle('Statistical Triangulation: Three-Condition Comparison', 
                    fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        output_file = self.output_dir / 'triangulation_comparison.png'
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization to: {output_file}")
        
    def run(self) -> Dict:
        """Run triangulation statistical significance testing."""
        logger.info("="*60)
        logger.info("Starting Triangulation Statistical Testing")
        logger.info("="*60)
        
        # Load all results
        baseline_data, targeted_results, zero_disc_results = self.load_all_results()
        
        # Extract baseline metrics
        baseline_metrics = self.extract_baseline_metrics(baseline_data)
        logger.info(f"Baseline: {baseline_metrics['n_correct']} correct, "
                   f"{baseline_metrics['n_incorrect']} incorrect")
        
        # Perform correction triangulation
        logger.info("\n" + "="*40)
        logger.info("CORRECTION TRIANGULATION")
        logger.info("="*40)
        correction_triangulation = self.perform_correction_triangulation(
            baseline_data, targeted_results, zero_disc_results
        )
        
        # Perform corruption triangulation
        logger.info("\n" + "="*40)
        logger.info("CORRUPTION TRIANGULATION")
        logger.info("="*40)
        corruption_triangulation = self.perform_corruption_triangulation(
            baseline_data, targeted_results, zero_disc_results
        )
        
        # Generate interpretation
        interpretation = self.interpret_triangulation(
            correction_triangulation, corruption_triangulation
        )
        
        # Create visualization
        self.create_visualization(correction_triangulation, corruption_triangulation)
        
        # Prepare comprehensive results
        results = {
            'metadata': {
                'phase': '4.14',
                'description': 'Statistical triangulation testing for PVA steering validation',
                'test_type': 'binomial',
                'alternative': 'greater',
                'significance_level': self.alpha,
                'timestamp': datetime.now().isoformat()
            },
            'baseline_metrics': baseline_metrics,
            'triangulation_results': {
                'correction': correction_triangulation,
                'corruption': corruption_triangulation
            },
            'interpretation': interpretation,
            'summary': {
                'validation_strength': interpretation['validation_strength'],
                'validity_checks': interpretation['validity_checks'],
                'correction_p_values': {
                    'baseline_vs_targeted': correction_triangulation['comparisons']['baseline_vs_targeted']['p_value'],
                    'zero_disc_vs_targeted': correction_triangulation['comparisons']['zero_disc_vs_targeted']['p_value'],
                    'baseline_vs_zero_disc': correction_triangulation['comparisons']['baseline_vs_zero_disc']['p_value']
                },
                'corruption_p_values': {
                    'baseline_vs_targeted': corruption_triangulation['comparisons']['baseline_vs_targeted']['p_value'],
                    'zero_disc_vs_targeted': corruption_triangulation['comparisons']['zero_disc_vs_targeted']['p_value'],
                    'baseline_vs_zero_disc': corruption_triangulation['comparisons']['baseline_vs_zero_disc']['p_value']
                },
                'effect_sizes': {
                    'correction': {
                        'baseline_to_targeted': correction_triangulation['comparisons']['baseline_vs_targeted']['effect_size'],
                        'zero_disc_to_targeted': correction_triangulation['comparisons']['zero_disc_vs_targeted']['effect_size'],
                        'baseline_to_zero_disc': correction_triangulation['comparisons']['baseline_vs_zero_disc']['effect_size']
                    },
                    'corruption': {
                        'baseline_to_targeted': corruption_triangulation['comparisons']['baseline_vs_targeted']['effect_size'],
                        'zero_disc_to_targeted': corruption_triangulation['comparisons']['zero_disc_vs_targeted']['effect_size'],
                        'baseline_to_zero_disc': corruption_triangulation['comparisons']['baseline_vs_zero_disc']['effect_size']
                    }
                }
            }
        }
        
        # Save results
        output_file = self.output_dir / 'triangulation_analysis.json'
        save_json(results, output_file)
        logger.info(f"Saved results to: {output_file}")
        
        # Save summary report
        self._save_summary_report(results)
        
        # Log summary
        logger.info("\n" + "="*60)
        logger.info("TRIANGULATION RESULTS")
        logger.info("="*60)
        logger.info(f"Validation Strength: {interpretation['validation_strength']}")
        logger.info("")
        logger.info("Correction Rates:")
        logger.info(f"  Baseline: {correction_triangulation['rates']['baseline']:.2%}")
        logger.info(f"  Zero-disc: {correction_triangulation['rates']['zero_discrimination']:.2%}")
        logger.info(f"  Targeted: {correction_triangulation['rates']['targeted']:.2%}")
        logger.info("")
        logger.info("Corruption Rates:")
        logger.info(f"  Baseline: {corruption_triangulation['rates']['baseline']:.2%}")
        logger.info(f"  Zero-disc: {corruption_triangulation['rates']['zero_discrimination']:.2%}")
        logger.info(f"  Targeted: {corruption_triangulation['rates']['targeted']:.2%}")
        logger.info("")
        logger.info("Validity Checks:")
        for check, passed in interpretation['validity_checks'].items():
            logger.info(f"  {check}: {'✓ PASS' if passed else '✗ FAIL'}")
        logger.info("")
        logger.info(f"Conclusion: {interpretation['interpretation']}")
        logger.info("="*60)
        
        return results
        
    def _save_summary_report(self, results: Dict) -> None:
        """Save human-readable summary report."""
        report_lines = [
            "PHASE 4.14: STATISTICAL TRIANGULATION ANALYSIS",
            "=" * 60,
            "",
            f"Generated: {results['metadata']['timestamp']}",
            f"Significance Level: {self.alpha}",
            "",
            "TRIANGULATION APPROACH",
            "-" * 40,
            "This analysis compares three conditions:",
            "1. Baseline: No steering (raw model generation)",
            "2. Zero-Discrimination: Steering with random non-discriminative features",
            "3. Targeted: Steering with discriminative PVA features",
            "",
            "Three pairwise comparisons validate:",
            "- Baseline vs Targeted: Does steering work?",
            "- Zero-disc vs Targeted: Is effect feature-specific?",
            "- Baseline vs Zero-disc: Are controls valid?",
            "",
            "CORRECTION EXPERIMENTS (Incorrect→Correct)",
            "-" * 40
        ]
        
        corr_tri = results['triangulation_results']['correction']
        report_lines.extend([
            f"Baseline Rate: {corr_tri['rates']['baseline']:.2%}",
            f"Zero-disc Rate: {corr_tri['rates']['zero_discrimination']:.2%}",
            f"Targeted Rate: {corr_tri['rates']['targeted']:.2%}",
            "",
            "Statistical Tests:",
            f"  Baseline→Targeted: p={corr_tri['comparisons']['baseline_vs_targeted']['p_value']:.2e} "
            f"{'(SIGNIFICANT)' if corr_tri['comparisons']['baseline_vs_targeted']['significant'] else '(not significant)'}",
            f"  Zero-disc→Targeted: p={corr_tri['comparisons']['zero_disc_vs_targeted']['p_value']:.2e} "
            f"{'(SIGNIFICANT)' if corr_tri['comparisons']['zero_disc_vs_targeted']['significant'] else '(not significant)'}",
            f"  Baseline→Zero-disc: p={corr_tri['comparisons']['baseline_vs_zero_disc']['p_value']:.2e} "
            f"{'(SIGNIFICANT)' if corr_tri['comparisons']['baseline_vs_zero_disc']['significant'] else '(not significant)'}",
            "",
            f"Interpretation: {results['interpretation']['detailed_findings']['correction']}",
            "",
            "CORRUPTION EXPERIMENTS (Correct→Incorrect)",
            "-" * 40
        ])
        
        corr_tri = results['triangulation_results']['corruption']
        report_lines.extend([
            f"Baseline Rate: {corr_tri['rates']['baseline']:.2%}",
            f"Zero-disc Rate: {corr_tri['rates']['zero_discrimination']:.2%}",
            f"Targeted Rate: {corr_tri['rates']['targeted']:.2%}",
            "",
            "Statistical Tests:",
            f"  Baseline→Targeted: p={corr_tri['comparisons']['baseline_vs_targeted']['p_value']:.2e} "
            f"{'(SIGNIFICANT)' if corr_tri['comparisons']['baseline_vs_targeted']['significant'] else '(not significant)'}",
            f"  Zero-disc→Targeted: p={corr_tri['comparisons']['zero_disc_vs_targeted']['p_value']:.2e} "
            f"{'(SIGNIFICANT)' if corr_tri['comparisons']['zero_disc_vs_targeted']['significant'] else '(not significant)'}",
            f"  Baseline→Zero-disc: p={corr_tri['comparisons']['baseline_vs_zero_disc']['p_value']:.2e} "
            f"{'(SIGNIFICANT)' if corr_tri['comparisons']['baseline_vs_zero_disc']['significant'] else '(not significant)'}",
            "",
            f"Interpretation: {results['interpretation']['detailed_findings']['corruption']}",
            "",
            "OVERALL VALIDATION",
            "-" * 40,
            f"Validation Strength: {results['interpretation']['validation_strength']}",
            "",
            "Validity Checks:"
        ])
        
        for check, passed in results['interpretation']['validity_checks'].items():
            report_lines.append(f"  {check}: {'✓ PASS' if passed else '✗ FAIL'}")
        
        report_lines.extend([
            "",
            "CONCLUSION",
            "-" * 40,
            results['interpretation']['interpretation'],
            "",
            "=" * 60
        ])
        
        report_file = self.output_dir / 'triangulation_summary.txt'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Saved summary report to: {report_file}")