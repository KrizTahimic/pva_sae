# Refactor TODO

## Preamble: Is This Worth It?

- [ ] Is this worth it before doing future_directions.md and iclr_reviewers_feedback.md?
    - [ ] Pros: I want to learn how to code and setup codebase better. It seems to be a gift that keeps on giving.
    - [ ] Cons: This may consume a lot of time. Better to use it to implement other experiments. Most likely it will be ugly again after adding those experiments anyway.
    - [ ] Maybe do something in the middle. Do only the low effort, high reward.
- [ ] Plan intensively with Claude. Create new file or just redo this?
- [ ] Is there other improvements/refactor I need to consider? Ask Claude.
- [ ] I want to write code better and have better design foresight on what I'm about to do. Do this while on learning mode I guess. Also have learning_notes.md while doing this.

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
