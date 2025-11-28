# Language Models with Checkpoints, SAEs, and MBPP Results

**Pythia is the only model family meeting all three criteria**, though the intersection reveals a significant gap in the open-source AI ecosystem. After comprehensive research across academic papers, model repositories, and technical documentation, only EleutherAI's Pythia suite definitively has publicly available training checkpoints, Sparse Autoencoders, and MBPP benchmark results—and even then, MBPP performance is modest due to Pythia not being code-specialized.

## Pythia: The sole model meeting all criteria

**EleutherAI's Pythia suite** was explicitly designed for interpretability research and remains the only model family with all three components publicly accessible.

**Training Checkpoints:**
- **154 checkpoints per model** at steps 0 (initialization), 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1000, then every 1,000 steps
- **8 model sizes**: 70M, 160M, 410M, 1B, 1.4B, 2.8B, 6.9B, 12B parameters
- **16 total models**: Standard Pile and deduplicated Pile variants
- **Access**: HuggingFace branches via `revision="step3000"` syntax
- **Link**: https://huggingface.co/EleutherAI (all Pythia models)
- **GitHub**: https://github.com/EleutherAI/pythia

**Sparse Autoencoders:**
- **EleutherAI SAE Collection**: https://huggingface.co/collections/EleutherAI/sparse-autoencoders
- Specific models include `EleutherAI/sae-pythia-160m-32x` with multiple expansion factors
- **Samuel Marks' dictionaries**: SAEs for Pythia-70m-deduped covering MLP outputs, attention outputs, and residual streams at https://baulab.us/u/smarks/autoencoders/
- **Multi-layer SAEs**: `tim-lawson/mlsae` trained on Pythia-70m-deduped
- **SAELens integration**: Full support for loading and training Pythia SAEs
- **Neuronpedia**: Interactive feature exploration available

**MBPP Results:**
| Model | MBPP Pass@1 | Source |
|-------|-------------|--------|
| Pythia-12B | **17.8%** | arXiv:2403.04811 |
| Pythia-12B (decontaminated) | 15.8-17.0% | Same study |

The relatively low MBPP score reflects that Pythia was trained on The Pile (general text corpus), not code-specific data. For comparison, StarCoderBase-15.5B achieves **41.6%** on the same benchmark.

## Why other promising models fall short

Several models nearly meet all criteria but have critical gaps:

**OLMo (AI2)** offers exceptional openness with **500+ training checkpoints** per model (1B to 32B sizes) and confirmed MBPP results (**60.2%** for OLMo-3-32B), but **no publicly released SAEs exist**. The research paper arXiv:2503.06394 trained SAEs on OLMo checkpoints for internal study, but these weights aren't available.
- Checkpoints: https://huggingface.co/allenai/OLMo-7B (use revision branches)

**Gemma 2 (Google)** has the most comprehensive SAE suite—**Gemmascope** covers 400+ SAEs across all layers of Gemma 2 2B, 9B, and 27B with both residual stream, MLP, and attention components. Gemma also has strong MBPP performance. However, **Google releases only final model weights**, not intermediate training checkpoints.
- SAEs: https://huggingface.co/google/gemma-scope
- Interactive demo: https://neuronpedia.org/gemma-scope

**Llama 3.x (Meta)** has extensive SAE coverage through **Llama Scope** (256 SAEs on Llama-3.1-8B-Base) and Goodfire's research, plus strong MBPP scores (**87-88%** for Llama 3.3 70B). However, Meta **does not release training checkpoints**—a frequently requested feature in GitHub issues that remains unaddressed.
- SAEs: https://huggingface.co/fnlp/Llama-Scope

**TinyLlama** (1.1B parameters) has intermediate checkpoints at 500B, 1T, 1.5T, and 3T tokens, and MBPP evaluations exist. However, while SAE research demos use TinyLlama, **no formally released pre-trained SAE weights** are available in standard repositories.

## Summary of model coverage

| Model | Checkpoints | SAEs | MBPP | All Three? |
|-------|-------------|------|------|------------|
| **Pythia** | ✅ 154/model | ✅ Multiple | ✅ 17.8% | **YES** |
| OLMo | ✅ 500+/model | ❌ None | ✅ 60.2% | No |
| Gemma 2 | ❌ Final only | ✅ 400+ SAEs | ✅ Strong | No |
| Llama 3.x | ❌ Final only | ✅ Llama Scope | ✅ 87-88% | No |
| TinyLlama | ✅ Multiple | ⚠️ Research only | ✅ Evaluated | No |
| LLM360 Amber | ✅ 360 ckpts | ❌ None | ❌ None | No |
| GPT-2 | ❌ None | ✅ Extensive | ❌ N/A | No |
| BLOOM | ⚠️ Limited | ❌ None | ❌ None | No |

## Practical recommendations for researchers

For studying **feature emergence during training** with code generation capabilities, Pythia remains the only verified option. Researchers should access checkpoints via HuggingFace's revision parameter and load SAEs through SAELens or directly from EleutherAI's collection.

For those requiring **stronger code performance**, consider training your own SAEs on OLMo using the EleutherAI/sparsify library—OLMo provides checkpoints and achieves 60%+ MBPP, needing only SAE training investment.

The ecosystem gap is notable: models prioritizing **interpretability** (Pythia) release checkpoints but lack code specialization, while **performance-focused** models (Gemma, Llama) have strong benchmarks but closed training artifacts. This creates an opportunity for the community to train and release SAEs on OLMo, which would create a second model family meeting all three criteria with substantially better code generation performance.

## Conclusion

The intersection of training checkpoints, SAEs, and MBPP benchmarks is remarkably narrow—**only Pythia qualifies**. This reflects divergent priorities in model development: interpretability-focused releases versus performance-focused releases rarely overlap. For researchers needing all three components, Pythia's 70M-12B model family with 154 checkpoints each, multiple SAE implementations, and documented (if modest) MBPP performance represents the only currently viable option. OLMo presents the clearest path to expanding this set, requiring only community investment in SAE training to unlock a fully open, higher-performing alternative.