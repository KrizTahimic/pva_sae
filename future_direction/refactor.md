# Refactor TODO

## Preamble: Is This Worth It?

- [ ] Is this worth it before doing future_directions.md and iclr_reviewers_feedback.md?
    - [ ] Pros: I want to learn how to code and setup codebase better. It seems to be a gift that keeps on giving.
    - [ ] Cons: This may consume a lot of time. Better to use it to implement other experiments. Most likely it will be ugly again after adding those experiments anyway.
    - [ ] Maybe do something in the middle. Do only the low effort, high reward.
- [ ] Plan intensively with Claude. Create new file or just redo this?
- [ ] Is there other improvements/refactor I need to consider? Ask Claude.
- [ ] I want to write code better and have better design foresight on what I'm about to do. Do this while on learning mode I guess. Also have learning_notes.md while doing this.
- [ ] Should I use einops and einsum?

---

## Phase 0: Decisions (Must Decide First)

These decisions affect how you approach everything else.

- [ ] Is there a better way to handle my data? I used to like parquet more but now I find it easier to view json files. But let's discuss the things to consider.
- [ ] Do we implement tests or not necessary for this kind of codebase?
- [ ] Is there other codebase architecture I should consider?
- [ ] Why is the inspiration code so few? While mine is so long?

---

## Phase 1: Cleanup (No Dependencies, Enables Everything Else)

Quick wins that make the codebase easier to work with.

- [ ] Delete not needed md or files anymore.
- [ ] Add better folder names especially the folders with no names just phase number.
- [ ] Improve logging by a lot. Right now it's almost useless as you don't know where it is going and it's mixed up and sometimes it's working, sometimes not.

---

## Phase 2: Core Infrastructure (Foundation for Later Work)

Fix the plumbing before building on top.

- [ ] Fix why there are multiple things needed to add when adding a new phase in run.py multiple times, config.py
- [ ] Make my data be in HuggingFace not in folders! IMPORTANT. Major improvement.

---

## Phase 3: Common Module Refactor (Sequential Chain)

Do these in order - each step depends on the previous.

- [ ] Merge common and common_simplified
- [ ] Add other common/reused functions in common
    - [ ] Not only the things I already use that is just located in other phase files but also notice the other repeated functions throughout most of the phases. Or is this even a good decision because sometimes it may constrain us. Flexibility is also a trait we want in some instances.
- [ ] Have better categorization for common
- [ ] Make/create a checkpointing function as a wrapper something so I don't need to reimplement it every phase? Is this possible? What is the design?

---

## Phase 4: Code Quality (Depends on Phase 3)

Polish the code after the structure is stable.

- [ ] Have consistent variable naming especially with the data.
- [ ] Write code better? More like Karpathy. Like list comprehension.

---

## Phase 5: Polish & Extras (After Core Refactoring)

Nice-to-haves once the foundation is solid.

- [ ] Improve notebooks. Remove unnecessary cells. Also do list comprehensions.
    - [ ] Understand matplotlib and pandas logic or how it works.
- [ ] Fix the figure generation code. Currently it looks soooo messy.
    - [ ] Make all figures correction green, corruption red, and pick a color for preservation.
- [ ] Add here the ICML LaTeX.

---

## Independent (Anytime)

Can be done in parallel with any phase.

- [ ] Consider other improvements to Claude Code like skills.md or hooks to improve my workflow.
    - [ ] Install Claude Code marketplace.

---

## ICML Submission Tasks (Based on ICLR Reviewer Feedback)

Address reviewer concerns with minimal compute.

- [ ] **Top-10 features table + Feature-Selection Landscape scatter plot** (Reviewers 7JAK, jwL5)
    - [ ] Create table showing top-10 features per direction with separation scores, t-statistics, and AUROC
    - [ ] Create scatter plot of separation scores vs t-statistics to show chosen features are statistical outliers (>3σ from mean)
    - Response: "We added Table X showing the top-10 features and Figure Y visualizing the feature landscape. Our top features are clear statistical outliers (>3σ from mean). We focus on top-1 as steering experiments are computationally expensive; examining multiple features is valuable future work."
    - **Decision:** Skip full steering on top-5 features - prediction metrics + visualization sufficient to address concern

- [x] **LLAMA + HumanEval experiments** (All reviewers) - IN PROGRESS
    - Addresses "single model, single benchmark" criticism
    - HumanEval transfer results already show strong generalization (F1: 0.821 → 0.920 for incorrect-predicting)

- [x] **Language-agnostic directions** (Reviewers vRko, jwL5) - SKIP
    - **Decision:** Not worth the effort. Reviewers said "ideally" not "required"
    - Would require HumanEval-X, uncertain SAE behavior on other languages
    - With MBPP + HumanEval + LLAMA, we have 3 evaluation points already
    - Frame as future work: "Investigating language-agnostic directions across programming languages is an important direction for future work."

- [x] **Difficulty-agnostic directions** (Reviewer 7JAK) - SKIP
    - **Decision:** Current difficulty stratification (cyclomatic complexity bucketing) is flawed
    - All MBPP/HumanEval problems are beginner-level - artificial bucketing not meaningful
    - Model can't solve harder benchmarks (APPS, CodeContests) so can't study correctness on them
    - Frame honestly: "Our study focuses on entry-level programming tasks where the model achieves non-trivial performance (~30% pass rate). Generalization to more complex programming challenges is limited by current model capabilities."
