# Phase 2 Improvements TODO

## Context
Phase 2 has been updated to implement global feature selection across all layers, selecting the top 20 features globally rather than just the best feature per layer. This aligns with the SAE analysis specification.

## Phase 3.5 Compatibility Issues

### Current State
- Phase 3.5 expects a single "best_layer" from Phase 2 (via `best_layer.json`)
- It extracts activations from only this one layer for temperature robustness testing
- The `temperature_test_layer` is currently hardcoded in the config

### After Phase 2 Update
- Phase 2 now outputs top 20 features that may come from multiple layers
- No single "best_layer" is identified
- Per-layer feature rankings are saved in `layer_{idx}_features.json`

## Future Improvements Needed

### Option 1: Multi-Layer Extraction (Recommended)
- Modify Phase 3.5 to extract activations from all layers that contribute to top 20
- Store activations for multiple layers per task
- Update analysis to use features from their respective layers

### Option 2: Best Layer Heuristic
- Define "best layer" as the layer contributing most features to top 20
- Create a new output file with this heuristic
- Minimal changes to Phase 3.5

### Option 3: Feature-Specific Analysis
- Completely redesign Phase 3.5 to analyze individual features
- Test temperature robustness per feature rather than per layer
- Most accurate but requires significant refactoring

## Additional Considerations

### Pile Filtering Integration
- The specification mentions filtering features that activate >2% on generic text
- This will further reduce the feature set and may change layer distribution
- Phase 3.5 should be designed to handle dynamic feature sets

### Performance Implications
- Multi-layer extraction will increase computation time
- Consider caching strategies for frequently accessed layers
- May need to parallelize extraction across layers

## Recommended Next Steps

1. **Immediate**: Add a temporary compatibility layer that selects "best layer" based on top feature distribution
2. **Short-term**: Implement Option 1 (multi-layer extraction) for more accurate analysis
3. **Long-term**: Design Option 3 (feature-specific analysis) when pile filtering is added

## Code Locations to Update

- `phase3_5_temperature_robustness/temperature_runner.py`: Main runner logic
- `common/config.py`: Remove hardcoded `temperature_test_layer`
- New utility needed: Feature-to-layer mapping for efficient extraction