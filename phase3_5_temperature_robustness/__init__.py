"""
Phase 3.5: Temperature robustness testing.

This phase runs AFTER Phase 2 has identified the best layer containing PVA latent directions.
It generates code solutions at multiple temperatures for the validation split,
extracting activations only from the single best layer identified in Phase 2.
"""