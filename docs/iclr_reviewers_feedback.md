# ICLR 2026 Reviews - Submission 8419

# Things to Code
- [x] **Selective steering implementation** (Reviewers RXZd, vRko)
  - [x] Apply selective steering approach to reduce corruption rate
  - (consider a different approach or conclusion) Like in its current form selective steering is not advisable also. Better to do other strategy maybe like do no steering first then only if you already check the generated code is wrong then in the rerun or retry of code generation the steering will be activated.

- [ ] **LLAMA and HumanEval extension** (Reviewers RXZd, vRko, 7JAK, jwL5)
  - [ ] Perform Mechanistic Analysis with `HumanEval`
  - [ ] Run all tests on `meta-llama/Llama-3.1-8B` and `meta-llama/Llama-3.1-8B-Instruct` with `llama_scope_lxr_8x`
  - [ ] Perform all Mechanistic Analysis
  - **SAE Verified**: `fnlp/Llama-Scope` 32K (8x expansion) SAE matches Neuronpedia's `llamascope-res-32k`. Confirmed via [Neuronpedia Llama Scope](https://www.neuronpedia.org/llama-scope) which explicitly references `fnlp/Llama3_1-8B-Base-LXR-8x`. The 128K (32x) variant is NOT recommended due to many inactive features.

- [ ] **Feature threshold sensitivity analysis** (Reviewer RXZd)
  - [ ] Test sensitivity to the >2% activation threshold on pile-10k
  - [ ] Report how many features get filtered out in top 25

- [ ] **Feature-Selection Landscape visualization** (Reviewer jwL5)
  - [ ] Create scatter plot of separation scores and t-statistics to show chosen features are outliers

- [ ] **Steering coefficient search plots** (Reviewer 7JAK)
  - [ ] Add plots showing steering coefficient search process to appendix

# Things to Fix

- [ ] **Add missing references** (Reviewer vRko)
  - [ ] Sparse Autoencoders Find Highly Interpretable Features in Language Models
  - [ ] A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models

- [ ] **Reframe SAE polysemanticity discussion** (Reviewer RXZd)
  - [ ] Reword statement to make it clear we don't claim to solve polysemanticity
  - [ ] Better frame that SAE is a current proposed solution that may be improved

- [ ] **Clarify detection direction application in practical settings** (Reviewer vRko)
  - [ ] Explain how detection direction can be combined with existing static analysis tools
  - [ ] Describe first line of defense approach, potentially hidden from user but applied by LLM provider

- [ ] **Improve methodology clarity** (Reviewer 7JAK)
  - [ ] Clarify indexing of equations 5-7
  - [ ] Clarify classification setup using latent activations (AUROC and F1 scores explanation)
  - [ ] Change terminology from "feature" to "latents" following Ferrando et al., 2024
  - [ ] Add steering coefficient search plots in appendix

- [ ] **Fix Figure 3 values** (Reviewer 7JAK)
  - [ ] Replace placeholder graph with actual data that matches text values

- [ ] **Minor fixes** (Reviewer 7JAK)
  - [ ] Define MBPP in line 56 (not just line 140)
  - [ ] Fix Elhage et al. 2022 reference link (currently incomplete, only goes to "/toy")

- [ ] **Add theoretical grounding citations** (Reviewer 7JAK)
  - [ ] Donoho, D.L. (2006). Compressed sensing
  - [ ] Thorpe, S.J. (1989). Local vs. Distributed Coding
  - [ ] Bengio, Y., et al. (2013). Representation learning
  - [ ] Arora, S., et al. (2018). Linear algebraic structure of word senses
  - [ ] Goh, G. (2016). Decoding The Thought Vector
  - [ ] Olah, C., et al. (2020). Zoom In: An Introduction to Circuits
  - [ ] Elhage, N., et al. (2022). Softmax Linear Units

- [ ] **Add SAE limitations discussion** (Reviewer 7JAK)
  - [ ] Dead features
  - [ ] Imperfect reconstruction
  - [ ] Evaluation challenges
  - [ ] Imperfect disentanglement
  - [ ] Add references: Gao & Dupré la Tour (2024), Bricken et al. (2023)

- [ ] **Improve methodology documentation** (Reviewer jwL5)
  - [ ] Add pipeline details in appendix
  - [ ] Define S (similarity) metric clearly (tokenizer, reference, surface vs AST, averaging window)
  - [ ] Clarify "initially correct" labeling, accuracy, and sample counts
  - [ ] Reference Figure 7 in the text

# Things to Defend

- [ ] **Asymmetric finding (incorrect vs correct detection)** (Reviewers RXZd, vRko, 7JAK)
  - [ ] Frame as interesting asymmetric finding revealing fundamental insights
  - [ ] Negative mechanism identification failure is a legitimate research finding

- [ ] **Error type breakdown** (Reviewer RXZd)
  - [ ] Respond with inspiration paper's rebuttal: "The primary objective is not to create perfect distinctions... minor classification errors are acceptable, as they don't impact the core conclusions"

- [ ] **Claude usage in methodology** (Reviewer vRko)
  - [ ] Clarify didn't use Claude for methodology itself (used Gemma)
  - [ ] Used Claude for code generation but with specific instructions and verification
  - [ ] Literature review was self-conducted, Claude only helped find missed sources
  - [ ] Writing: made conclusions/findings, Claude drafted rough outline which was refined

- [ ] **Generalization to multi-language/multi-file projects** (Reviewer vRko)
  - [ ] Beyond scope - focus is on Python
  - [ ] Suspect same methodology could work with multi-language/multifile dataset

- [ ] **Top 5-10 latent features robustness** (Reviewer 7JAK)
  - [ ] Explain feature selection rationale
  - [ ] Note steering testing is costly (requires separate runs)
  - [ ] Clarify why specific features were chosen

- [ ] **Control feature L1-4801 selection** (Reviewer 7JAK)
  - [ ] Followed Ferrando et al., 2024 methodology (latent with zero separation score)

- [ ] **Test cases prompt suggestion practicality** (Reviewer 7JAK)
  - [ ] Work is primarily mechanistic interpretability research
  - [ ] Acknowledge practical limitations in real-world scenarios
  - [ ] Suggestion is for prompts, not LLM improvement

- [ ] **Paper contribution and scope** (Reviewer 7JAK)
  - [ ] Fundamentally mechanistic interpretability research
  - [ ] Core contribution: discovering disentangled code correctness representations exist
  - [ ] Not proposing new methods to achieve disentanglement
  - [ ] Not claiming to solve entanglement problem

- [ ] **SAE entanglement problem** (Reviewer 7JAK)
  - [ ] Don't claim SAEs solve entanglement
  - [ ] Using SAEs as established tool with known limitations
  - [ ] Contribution is applying existing tools, not solving their fundamental limitations

- [ ] **"Causal influence" terminology** (Reviewer 7JAK)
  - [ ] Maintain term to distinguish from correlational metrics
  - [ ] Perform interventions to test causal influence, not merely to intervene
  - [ ] Emphasizes establishing causality beyond correlation

- [ ] **SAE symbol connections** (Reviewer 7JAK)
  - [ ] Ask reviewer to specify which symbols need clarification
  - [ ] Note structured progression: intro → Section 2 formal notation → methodology application

- [ ] **Generalization across code complexity levels** (Reviewer 7JAK)
  - [ ] Study focuses on specific scope: problems solvable by Gemma-2-2B (MBPP ~30% pass rate)
  - [ ] Acknowledge generalization is important future work
  - [ ] Findings establish representations exist in this domain

- [ ] **Control condition weakness** (Reviewer jwL5)
  - [ ] Inspiration paper used 1 control for steering, 10 for attention analysis
  - [ ] Single control is acceptable following established methodology

## Review 1: Reviewer RXZd

**Date:** 04 Nov 2025, 16:46 (modified: 12 Nov 2025, 12:05)

### Summary

This paper applies sparse autoencoders (SAEs) to decompose LLM representations and identify directions corresponding to code correctness in the Gemma-2-2b model. The authors discover two distinct mechanisms: detection directions that reliably predict incorrect code (F1: 0.821) but fail as confidence indicators for correct code (F1: 0.504), and steering directions that achieve modest corrections (4.04% of errors fixed) while corrupting 14.66% of initially correct code. Through mechanistic analysis including activation steering, attention weight analysis, and weight orthogonalization, the authors demonstrate that successful code generation depends primarily on attending to test cases rather than problem descriptions, and that correct-steering directions are causally necessary for code generation. The work represents the first application of SAEs to address superposition in code representations and suggests practical applications including using predictor directions as error alarms and applying selective rather than constant steering interventions.

### Scores

- **Soundness:** 3 (good)
- **Presentation:** 3 (good)
- **Contribution:** 3 (good)
- **Rating:** 2 (reject, not good enough)
- **Confidence:** 4 (You are confident in your assessment, but not absolutely certain)

### Strengths

- First use of SAEs to decompose code correctness representations, extending entity recognition frameworks to a new domain with appropriate adaptations.
- The paper employs multiple validation techniques (steering, attention analysis, weight orthogonalization) that provide converging evidence for causal mechanisms rather than mere correlations.
- The discovery that models encode incorrect code as detectable anomalies but lack corresponding representations for correctness reveals fundamental insights about how LLMs represent code validity.
- Demonstrating that code correctness mechanisms persist from base to chat models (F1: 0.821 → 0.772 for error detection) suggests interpretability methods can generalize across training stages.

### Weaknesses

- The entire analysis is conducted on a single model (Gemma-2-2b, 2B parameters) and single benchmark (MBPP). This raises serious questions about whether findings generalize to other model families, sizes, or programming tasks (e.g., HumanEval, APPS, more complex algorithms). The authors should at minimum validate key findings on one additional model and benchmark.
- The 4.04% correction rate is overshadowed by the 14.66% corruption rate which is nearly 4-fold difference. While the authors acknowledge this necessitates "selective steering," they don't demonstrate or evaluate such a selective approach. The practical utility remains unclear without showing that combining detection and steering actually improves overall performance.
- The incorrect steering direction identified via separation scores produces only repetitive '8' tokens and achieves merely 2.2% corrections (below control at 5.5%). This represents a fundamental failure of the separation score methodology for identifying incorrect features.
- The discovery that the incorrect-predicting feature activates on both anomalous code patterns AND foreign language tokens directly contradicts the SAE promise of monosemantic features. The paper dismisses this as "empirical evidence that SAEs do not fully solve polysemanticity" but doesn't adequately discuss implications for the reliability of their findings or alternative approaches.

### Questions

- Refer to weaknesses
- You exclude features activating >2% on pile-10k. How sensitive are results to this threshold?
- Do detection and steering directions work differently for different error types (syntax errors, logic errors)? Breaking down performance by error category would be valuable.

### Meta

- **Flag for Ethics Review:** No ethics review needed
- **Code of Conduct:** Yes

### My (Kriz) notes:
Summary
- The reviewer were able to objectively summarize my paper along with its results/data.

Scores
- The reviewer gave me the highest rubric but give the lowest rating. I think he thinks "This is good work (hence the 3s/4), but these specific weaknesses are too significant for ICLR acceptance (hence the 2/10)""

Strengths
- Notable takeaway is the asymmetry "- The discovery that models encode incorrect code as detectable anomalies but lack corresponding representations for correctness reveals fundamental insights about how LLMs represent code validity."

Weaknesses
- [Code] I do all(?) the test in `meta-llama/Llama-3.1-8B-Instruct` with `llama_scope_lxr_8x`
    - I plan to do Mechanistic Analysis with `HumanEval`
    - Maybe in AUROC, F1, and steering test only?
- [Code] I will apply selective steering
- [Defend] Right now is an interesting asymmetric finding. 
- [Defend&Fix] (Try to frame this better so they will be nicer to us) We use SAE knowing it doesn't fully solve superposition. It's one of the current proposed solutin that may be improved. While doing this work there are new tools like croscoders, transcoders.

Also in the future we hope this found issue can still be further improved in future decomposing tools like SAE. but for now we may suggest that still check even in false positives just to be more safe.

Superposition is still an ongoing field that has just recently been discovered. Decomposing is an active research. But here we show that current state can already be set an example as for practical application.

Reword the statemetn make it clear we don't calim to solve polysemanticity

Questions
- Refer to the notes above.
- [Code] I can do this but it will involve more coding. I can answer how many did get filter out. I need to answer this. On top 25 this is how many the is filtered out
- [Defend] Good thing to do. Should I do this? Seem complex to implement. I can also reponse what the inspiration say in their rebuttal "The primary objective is not to create perfect distinctions... minor classification errors are acceptable, as they don't impact the core conclusions."

---

## Review 2: Reviewer vRko

**Date:** 04 Nov 2025, 11:13 (modified: 12 Nov 2025, 12:05)

### Summary

This paper investigates the "code correctness" mechanism of Large Language Models (LLMs) in code generation tasks, motivated by the widespread application of AI code generation but with inherent reliability risks, especially in high-risk domains. The authors decompose the internal representation of LLMs using a Sparse Autoencoder (SAE) to identify directions relevant to code correctness: detected directions are used to predict errors, while manipulated directions can be used to correct them. The paper systematically analyzes the performance, attention distribution, and persistence of these mechanisms after model fine-tuning, finding that test cases are more critical than problem descriptions, and offers implications for practical development processes, such as improved prompt design and automatic error alerts.

### Scores

- **Soundness:** 2 (fair)
- **Presentation:** 2 (fair)
- **Contribution:** 2 (fair)
- **Rating:** 4 (marginally below the acceptance threshold, but would not mind if paper is accepted)
- **Confidence:** 3 (You are fairly confident in your assessment)

### Strengths

- This study elucidates the mechanism of using sparse autoencoders in code generation models, overcoming the interpretability challenge caused by feature superposition.
- It covers multiple aspects, including detection, manipulation, attention distribution, and weight orthogonalization, with rigorous experimental design and thorough statistical testing.
- It proposes superior code suggestion strategies, error alerting mechanisms, and targeted model intervention recommendations, demonstrating practical engineering value.
- It proves that the correctness mechanism in the pre-trained model remains effective after fine-tuning, possessing both theoretical significance and practical applicability.

### Weaknesses

- The correction rate was only 4.04%, while erroneous intervention resulted in 14.66% of correct code being corrupted, indicating a high risk in the practical application of the manipulation direction and insufficient exploration of how to reduce side effects.
- Although key experiments were conducted, the experiments were not comprehensive enough:
  1. All experiments were based on Gemma-2 and MBPP, failing to cover other models or more complex code scenarios, and the generalizability of the results needs further verification. *
  2. Negative mechanism identification failed: the "erroneous code" direction was not effectively identified, and the related experimental results and analysis were relatively weak.
- The pipelines relied on closed-source large models: Although manually reviewed, some experiments and literature searches relied on Claude, which may affect independence.
- Missing some references:
  - Sparse Autoencoders Find Highly Interpretable Features in Language Models
  - A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models
  - Sparse Autoencoders Find Highly Interpretable Features in Language Models

### Questions

- Can the features decomposed by a sparse autoencoder be generalized to other coding tasks (such as multi-language, multi-file projects)?
- How can the manipulation direction be optimized to reduce the rate of breaking correct code and make it more suitable for practical deployment?
- In actual development processes, how can the detection direction be combined with existing static analysis tools to improve code quality?
- The paper failed to effectively identify the "erroneous code" direction, are there alternative methods or future plans?

### Meta

- **Flag for Ethics Review:** No ethics review needed
- **Code of Conduct:** Yes

### My (Kriz) notes:

Summary
- Seem AI generated but fair summary of my research

Scores
- Can aim to increase the reviewers rating to 6 into acceptance threshold.

Strengths
- The reviewer liked the rigor in methodology (but not enough)

Weaknesses
- [Code] Do a test applying selected steering
-
    1. [Code] Do test in `meta-llama/Llama-3.1-8B-Instruct` with `llama_scope_lxr_8x` and `HumanEval`
    2. [Defend] Asymmetric fiding 
-  [Defend] I didn't use Claude for methodology perse. I used gemma. I used claude for generating code but I instruct specifically what and how it  should be implemented and I check every step. About the literature review, I conducted my own literature review and just prompted Claude to find sources I missed. For writing, I made the conclusion and findings, I just instruct Claude to write a draft of my rough outline which I later on refined further. I apologize if this is not made clearer in LLM Usage section before.
- [Fix] Thanks for pointing this out, Will add both of them in my references.

Questions
- [Defend] Beyond scope. We suspect we can use the same methodology just with a multi language, and multifile dataset then it could also work. 
    The focus of this research if on python
- [Code] Selective Steering
- [Fix] This belong in the first line of defense. Maybe even hidden from the user but applied by the LLM provider when generating code. Just repeat what I said to thesis
- [Defend&Fix] Use the predicting direction, or use other decomposing techniques like transcoders and crosscoders. Asymmetric finding.
---

## Review 3: Reviewer 7JAK

**Date:** 31 Oct 2025, 18:15 (modified: 12 Nov 2025, 12:05)

### Summary

The authors apply previously developed methods to analyze LLM performance in code generation tasks. This is a field that has seen widespread adoption as in both academia and industry the share LLM generated code is increasing. The authors applied sparse autoencoders on the residual streams of each layer to disentangle super-positioned representations and then detect the most impactful latent features. These were analyzed and intervened upon through a steering method. The authors confirmed previous results, that LLMs display anomaly detection mechanisms and not validity assessments. Furthermore, they found that applying steering directions to identified latent features leads to a trade-off between accurate code corrections and corrupting previously well generated code.

### Scores

- **Soundness:** 3 (good)
- **Presentation:** 2 (fair)
- **Contribution:** 2 (fair)
- **Rating:** 6 (marginally above the acceptance threshold, but would not mind if paper is rejected)
- **Confidence:** 4 (You are confident in your assessment, but not absolutely certain)

### Strengths

**a. Important and Timely Research Question**
- Addresses critical need for understanding LLM code generation reliability as these models enter industrial deployment
- Focuses on mechanistic interpretability, which is essential for trustworthy AI systems

**b. Novel Methodological Approach**
- Creative application of SAEs to decompose code correctness representations into interpretable features
- Addresses the superposition problem that compresses high-dimensional features into lower dimensions
- Uses multiple complementary analysis techniques (steering, attention, weight orthogonalization)

**c. Well-Structured Experimental Design**
- Systematic approach using t-statistics and separation scores
- Demonstrates both predictive capabilities and correction mechanisms
- Acknowledges trade-offs in the findings (error fixing vs. correct code preservation)

### Weaknesses

**a.** All of the analysis was performed on the latent features with highest metric as displayed in Table 1. Do the conclusions hold for say the top-5 to -10 latent features? Are the metric values for these features outliers compared to other features? Some further details in this aspect would significantly enhance the robustness of results.

**b.** There is some lack of clarity when reading the paper, particularly if unfamiliar with the methods developed in Ferrando et al., 2024 and Marks & Tegmark, 2023. There was some effort to mention details of the methods, although not enough was included to fully grasp the methods (In some cases it could better to simply guide the reader to the original paper). Some aspects that could be addressed include:

- Clarify indexing of equations 5-7
- Clarify the use of the identified latent feature for classification through thresholding, as understanding the appearance of AUROC and F1 scores was not clear.
- Personally, the usage of "feature" for the encoded position in the SAE can be error inducing as the word is more commonly used for input "features". Possibly using "latents" (as in Ferrando et al., 2024) could be better.
- Why was feature L1-4801 used as control?
- Some further clarification on how the steering strength parameter was obtained could be added as supplementary material.

**c.** The values plotted in Figure 3 generally do not seem to match the values mentioned in the text. The F1 value for incorrect-predicting at T=0 is below 0.8 while the reference value in the text is 0.821, and in line 246 the improvement 0.821->0.986 is mentioned whereas the values seem to lie in the [0.76, 0.83] interval.

**d.** The key suggestion given by the authors from the obtained results, is to include more test cases when prompting. While this is easily applicable in the used dataset for simple methods, it is often not so simple when developing methods in real scenarios. It would have a larger impact on SOTA if suggestions on improving LLMs to achieve validity assessment mechanisms.

### Minor Suggestions

- MBPP is not defined in line 56, later defined in line 140
- The link of reference Elhage et al. 2022 is incomplete, while the text is present the link only goes to "/toy".

### Questions

Overall while the authors did a good job it misses the core question of how does the issue of entanglement be solved.

**Methodological Gaps**
- No clear guarantee of SAE uniqueness - how do authors ensure SAEs avoid the entanglement problem they claim to solve?
- Mean deviation process in prediction direction needs better motivation and explanation
- Missing details on how different values in Lines 261-265 are obtained

**Conceptual and Terminological Issues**
- "Causal influence" (L306) would be more accurately termed "intervention"
- Introduction could better connect SAE methodology symbols to the actual methodology section
- Some claims about superposition and feature compression need stronger theoretical grounding

**Incomplete Analysis**
- Limited discussion of potential failure modes or limitations of the SAE approach
- Insufficient exploration of how findings generalize across different code types and complexity levels

### Meta

- **Flag for Ethics Review:** No ethics review needed
- **Code of Conduct:** Yes

### My (Kriz) notes:

Summary
- The reviewer where able to put in a nice way my negative results as legit finding.

Scores
- This reviewer gave the highest rating of 6.

Strengths
- I appreciate the reviewers feedback. The reviewer is able to hit the main points of my research

Weaknesses
**a.** [Defend||Code] Agree that this is interesting to do. Even to add to that, to also examine how deep or what layer have the highest performance as currently higher layer has concrete related tokens like 7 or "QUEUE" or "METHODS." But currently this is costly to test for steering at it will involve separate runs.
This is the other features. THe goal is steer to correctness.
To clarify why i picked this.
**b.**
- [Fix] I can easily clarify/add what l,j,i mean
- [Fix] We thank the reviewer for highlighting this unclear exposition. We will clarify the classification setup: We use latent activations $a_{l,j}(x)$ as continuous scores to predict code correctness. For each latent, we threshold activations across a range (e.g., np.linspace(scores.min(), scores.max(), 100)) to produce binary predictions, then compare against ground-truth correctness labels (from code execution). AUROC measures discrimination across all thresholds via the ROC curve, while F1 score is computed at the optimal threshold. We will add this methodological explanation in Section [X] to clarify how latent features translate to classification metrics.
- [Fix] This has been a contention while writing the paper. We'll now make this update following your advice. Thank you!
- [Defend] We copied how Ferrando et al., 2024 selected their control, which is by finding the latent with zero separation score.
- [Fix] We'll add the steering coefficient search plots in the appendix

**c.** [Fix] Thanks for pointing this out. We are sorry for this mistake. We fail to replace the placeholder graph that if we recall correctly came a test run with a subset of the supposed data. Rest assured that the real graph follow the same trend which we will now update to the revised manuscript.

**d.** [Defend] We appreciate this perspective. We acknowledge that providing additional test cases has practical limitations in real-world scenarios with complex codebases or proprietary systems. Our work is primarily mechanistic interpretability research aimed at understanding how LLMs represent code correctness. With this knowledge, one way to help LLMS is provide is with more test cases. Is meant to be a practical application knowing the application limitation. For the Prompt not the LLM.

Minor Suggestions
- [Fix] We will make this update. Thank you!
- [Fix] Glad you picked this up. This was caused by LateX syntax error. It will now be fixed for the revised version.

Questions
- [Defend&Fix] "We appreciate this insightful comment. We agree that our paper could better clarify its primary contribution and scope. Our work is fundamentally mechanistic interpretability research - applying existing interpretability techniques (SAEs, linear probing, causal interventions) to the previously unexplored domain of code correctness in LLMs. The core contribution is discovering and validating that disentangled code correctness representations exist and can be causally identified, rather than proposing new methods to achieve disentanglement. We have revised the paper to more clearly position this as exploratory mechanistic analysis with the primary goal of advancing our understanding of how LLMs represent code correctness, while noting that developing comprehensive solutions for the entanglement problem would be valuable future work."

Methodological Gaps
-  [Defend] (Maybe combine this answer to question's answer as they address the same issue) "We appreciate the opportunity to clarify this point. We do not claim that SAEs solve the entanglement problem. Rather, we use SAEs as an established mechanistic interpretability tool to discover and analyze latent representations, acknowledging their known limitations. SAEs are imperfect and don't guarantee perfect disentanglement - this is a well-documented limitation in the literature (Anthropic, 2024; others). Our contribution is applying existing tools to understand code correctness representations, not proposing solutions to fundamental limitations of those tools."
- 

Conceptual and Terminologial Issues
- [Defend] We appreciate the reviewer's attention to terminology. However, we respectfully maintain "causal influence" as it distinguishes the scientific goal of our steering experiments from the correlational metrics used earlier. We perform interventions specifically to test whether features causally influence code generation, not merely to intervene. This framing emphasizes the contribution of establishing causality beyond correlation, which is central to our validation approach.
- [Defend||Fix] We appreciate the reviewer's attention to readability. Could the reviewer specify which symbols or connections need clarification? We structured the paper to introduce SAE concepts generally in the introduction for broader context, then provide the specific SAE architecture we use (JumpReLU-based) with formal notation in Section 2 (equations 1-4, defining $a(\mathbf{x})$, $\mathbf{W}_{\text{dec}}$, etc.), which is then applied throughout the Methodology. We can add explicit forward references if the reviewer found specific transitions unclear.
    - What does the reviewer mean by the SAE symbol
- [Fix] This comment is appreciated. We added those papers. We thank the reviewer for this suggestion. We will strengthen the theoretical grounding by adding citations to the mathematical and theoretical foundations of superposition: compressed sensing theory (Donoho, 2006) provides the framework for sparse signal recovery from lower-dimensional spaces; distributed representation theory (Thorpe, 1989; Bengio et al., 2013) explains why neural networks favor such encodings; and prior work on polysemantic embeddings (Arora et al., 2018) demonstrates linear superposition in word representations using sparse coding methods analogous to SAEs. We will also cite interpretability work on neural feature superposition (Goh, 2016; Olah et al., 2020; Elhage et al., 2022). We will revise to make these theoretical connections more explicit.

    **References to add:**
    - Donoho, D.L. (2006). Compressed sensing. IEEE Transactions on Information Theory, 52(4), 1289-1306. https://ieeexplore.ieee.org/document/1614066
    - Thorpe, S.J. (1989). Local vs. Distributed Coding. Intellectica, 8(2), 3-40. https://www.persee.fr/doc/intel_0769-4113_1989_num_8_2_873
    - Bengio, Y., Courville, A., & Vincent, P. (2013). Representation learning: A review and new perspectives. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(8), 1798-1828. https://ieeexplore.ieee.org/document/6472238
    - Arora, S., Li, Y., Liang, Y., Ma, T., & Risteski, A. (2018). Linear algebraic structure of word senses, with applications to polysemy. Transactions of the Association for Computational Linguistics, 6, 483-495. https://aclanthology.org/Q18-1034/
    - Goh, G. (2016). Decoding The Thought Vector. https://gabgoh.github.io/ThoughtVectors/
    - Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., & Carter, S. (2020). Zoom In: An Introduction to Circuits. Distill. https://distill.pub/2020/circuits/zoom-in/
    - Elhage, N., et al. (2022). Softmax Linear Units. Transformer Circuits Thread. https://transformer-circuits.pub/2022/solu/index.html



Incomplete Analysis
- [Fix] We thank the reviewer for this important suggestion. We will add a dedicated discussion of SAE limitations and potential failure modes, drawing on recent literature. Key limitations include: (1) **dead features** - some latents may fail to activate, wasting capacity (Gao & Dupré la Tour, 2024), (2) **imperfect reconstruction** - SAEs don't achieve zero reconstruction loss, leaving model variance unexplained (Gao & Dupré la Tour, 2024), (3) **evaluation challenges** - lack of ground truth makes feature interpretability assessment subjective (Gao & Dupré la Tour, 2024; Bricken et al., 2023), and (4) **imperfect disentanglement** - SAE features may still exhibit polysemanticity (Bricken et al., 2023).

**References to add:**
- Gao, L., & Dupré la Tour, T. (2024). Scaling and evaluating sparse autoencoders. arXiv:2406.04093. https://arxiv.org/abs/2406.04093
- Bricken, T., et al. (2023). Towards Monosemanticity: Decomposing Language Models With Dictionary Learning. Transformer Circuits Thread. https://transformer-circuits.pub/2023/monosemantic-features

- [Defend] "We appreciate this feedback. Our study focuses on a specific scope of code complexity - problems solvable by Gemma-2-2B, model within our compute budget, (MBPP tasks with ~30% pass rate). We acknowledge that generalization across different code complexity levels and problem types is an important direction for future work. Our findings establish that disentangled code correctness representations exist and are causally relevant in this domain; investigating how these representations scale to more complex code (harder algorithms, larger codebases, different programming paradigms) would be valuable next steps. We have added a discussion of this limitation in Section [X] and the conclusion."

I didn't consider differnt complexity and languages. I added experiments llama and eval
---

## Review 4: Reviewer jwL5

**Date:** 23 Oct 2025, 04:23 (modified: 12 Nov 2025, 12:05)

### Summary

This paper uses layer-specific sparse autoencoders on Gemma-2-2B's residual stream to identify "predictor" directions that distinguish correct vs incorrect code generations and "steering" directions that, when added at inference, bias the model's generation. The authors produce key findings on coding benchmark MBPP, the best "incorrect-predicting" feature serves as a modest error alarm (F1 = 0.82), while the "correct-predicting" feature is weak (F1 = 0.504). Steering with the "correctness" direction repairs ~4.0% of initially wrong solutions but also corrupts ~14.7% of initially correct ones. The authors also perform an attention analysis showing steering shifts focus towards the unit tests. The authors also suggest that the correct directions are causal for code correctness behaviour as orthogonalizing a "correct" direction degrades ~83.6% of previously correct solutions vs. ~19% for a neutral control. The authors found that these directions largely transfer from the base model to a chat-tuned variant, indicating some robustness of the discovered mechanisms.

### Scores

- **Soundness:** 2 (fair)
- **Presentation:** 2 (fair)
- **Contribution:** 2 (fair)
- **Rating:** 4 (marginally below the acceptance threshold, but would not mind if paper is accepted)
- **Confidence:** 4 (You are confident in your assessment, but not absolutely certain)

### Strengths

**Novelty:** This is the first mechanistic interpretability study specifically aimed at code correctness in LLMs, offering fresh insights into a relatively unexplored question that prompts opportunities for follow-up directions in the research community.

**Transfer Signal:** Some of the discovered directions carry over from base to chat-tuned model, hinting these are not pure overfits to one checkpoint.

### Weaknesses

**Methodology Ambiguity:** The paper's high-level recipe is clear, but many operational details are missing or scattered, which makes replication and interpretation difficult. In particular, several core steps are not specified precisely enough to rule out confounds or to let others reproduce the results end-to-end:

- **Definition of S (similarity):** "Mean Python token similarity %" is undefined (tokenizer, reference, surface vs AST notion of similarity, averaging window). Clarifying this would strengthen the understanding.
- **Evaluation bookkeeping:** How "initially correct" is labeled, accuracy, and sample counts per condition. Clarifying this would strengthen the claims.
- **Control condition:** The neutral control feature has not been explored and a single control is weak to support specificity claims.
- **Feature-Selection Landscape:** The current search/selection feels under-characterized. Please add a compact Feature-Selection Landscape to show the chosen directions are genuine outliers rather than hand-picked. These additions make selection auditable at a glance, demonstrate how exceptional the chosen features are, and strengthen this paper's evidence.

**Single Benchmark Scope:** All results are on MBPP (short Python tasks). To support this paper claims about code correctness more broadly, it would help to include at least one additional benchmark (HumanEval, APPS) and, ideally, also non-Python. Showing that the key findings (e.g. stronger error alarm, test-focused attention, and orthogonalization effects) transfer across datasets/languages will materially improve generality and impact.

### Questions

**Small Comment:** It would be great to reference Figure 7 in the text.

### Meta

- **Flag for Ethics Review:** No ethics review needed
- **Code of Conduct:** Yes

### My (Kriz) notes:

Summary
- Fair summary of results

Scores
- Can still be convinced to increase the rating

Strengths
- I’m surprised the reviewers liked base to instruct model repurposing finding

Weaknesses
- [Fix] Methodology Ambiguity - Add my pipeline in the appendix.
- [Fix] Definition of S (similarity) - I can simply detail what I did here
- [Fix] Clarify
- [Defend] Control condition - I don't quite understand this concern. The inspiration paper only use 1 for steering but suprisingly 10 for attention analysis. Or just rebut that this is still acceptable. But I think. Same reply with the rebuttal of top 5 features.
- [Code] feature-Selection Landscape - try if I can easily do a scatter plot of separation scores and t-statistic of 
- [Code] Single Benchmark Scope - I can do HumanEval (I'm currently considering also doing HumanEval-X (C++, Java, JavaScript, and Go)). The only constraint is time. Maybe do Java only? or maybe stick with HumanEval only. 
---

## Summary of Ratings

| Reviewer | Rating | Soundness | Presentation | Contribution |
|----------|--------|-----------|--------------|--------------|
| RXZd     | 2      | 3         | 3            | 3            |
| vRko     | 4      | 2         | 2            | 2            |
| 7JAK     | 6      | 3         | 2            | 2            |
| jwL5     | 4      | 2         | 2            | 2            |

**Average Rating:** 4.0 (marginally below acceptance threshold)


# IMPORTANT

1. Try to request for compute. Exclusive one GPU access unlike before which is need to be shared.