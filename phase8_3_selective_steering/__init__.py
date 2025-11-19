"""
Phase 8.3: Selective Steering Based on Threshold Analysis

Applies Phase 4.8 correct steering ONLY when the incorrect-predicting direction
(from Phase 3.8) exceeds the optimal threshold (15.5086).

This phase combines:
- Threshold checking: Layer 19, Feature 5441 (incorrect-predicting from Phase 2.10 + Phase 3.8)
- Steering intervention: Layer 16, Feature 11225 (correct-steering from Phase 2.5 + Phase 4.8)

Key Features:
- Real-time threshold checking during generation (Option A)
- Split testing: separate experiments for initially correct vs initially incorrect problems
- Selective intervention: only steer when activation > threshold
- Preserves Phase 3.5 baseline when activation ≤ threshold (no unnecessary generation)

Expected Outcomes:
- Maintain correction capability (comparable to Phase 4.8's 4.04%)
- Reduce corruption rate (vs Phase 4.8's 14.66%)
- Demonstrate practical value of predictor directions for selective intervention
"""

from .selective_steering_analyzer import SelectiveSteeringAnalyzer

__all__ = ['SelectiveSteeringAnalyzer']
