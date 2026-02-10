---
description: Run comprehensive multi-agent code review with verification
argument-hint: "[--scope <dir_or_file>] [--severity critical|high] [--skip-interview]"
---

Run a comprehensive code review of this project using a multi-agent architecture with built-in verification to eliminate false positives.

## Arguments
- `--scope <path>`: Optional. Limit review to specific directory or file (default: entire project)
- `--severity <level>`: Optional. Only report findings at this level or above (default: all)
- `--skip-interview`: Optional. Skip the interactive interview phase and just produce the report

User request: $ARGUMENTS

---

## PHASE 1 — REVIEW (Parallel Specialist Agents)

Spawn **5 parallel agents** using the Task tool (subagent_type: `Explore`). Each agent MUST cite exact `file_path:line_number` for every finding. Each agent should explore broadly within their domain — the goal is to find issues we DON'T already know about.

**IMPORTANT**: All 5 agents must be launched in a SINGLE message (parallel tool calls). Set thoroughness to "very thorough".

### Codebase Context (share with all agents)
This is a research project with 30+ phases for SAE-based analysis of code correctness in LLMs. Key areas: `common/` (shared infra), `phase*/` (phase runners), `tests/`, `run.py` (CLI). Read CLAUDE.md for architecture details. The project uses PyTorch, pandas, multi-GPU parallelization, and parquet/safetensors for data.

**Known bug tendencies** (for calibration — these are patterns that have appeared before, so the codebase may have more like them, but do NOT limit your search to only these):
- Prompts double-wrapped by PromptBuilder on data that already has prompts
- Statistical tests with degenerate parameters (e.g., `binomtest(p=0)`)
- Cross-GPU merge logic that compares scores across GPUs instead of per-problem
- Error handlers that assume success (return True) instead of failure
- Hardcoded paths instead of using discovery functions

### Agent A — Bug Hunter
**Mandate**: Find logic errors, correctness bugs, and semantic mistakes anywhere in the codebase. Think like an adversary trying to find inputs or states that would produce wrong results. Look at math, conditionals, data flow, state management, concurrency, and anything else that could be logically wrong. No category is off-limits.

### Agent B — Test Coverage Auditor
**Mandate**: Assess what's tested and what isn't. Find critical code paths, complex logic, and failure modes that lack test coverage. Evaluate whether existing tests actually test meaningful behavior or just pass trivially. Identify the highest-risk untested areas. Look at the full `tests/` directory and compare against production code.

### Agent C — Error Handling Reviewer
**Mandate**: Review how the codebase handles failures, edge cases, and unexpected states. Look at exception handling, fallback behavior, default values on error paths, resource cleanup, and any place where something could go wrong at runtime. Consider file I/O failures, GPU memory issues, malformed data, and missing dependencies between phases.

### Agent D — Consistency Checker
**Mandate**: Look for inconsistencies across the codebase. This includes naming mismatches between phases, API contracts that don't match between producer and consumer, duplicated logic that has diverged, configuration values hardcoded in multiple places, and any place where two parts of the code disagree about how something should work. Check terminology, data schemas, path conventions, and behavioral assumptions.

### Agent E — Performance & Resource Reviewer
**Mandate**: Find resource waste, memory leaks, and performance bottlenecks. This codebase runs multi-hour GPU jobs where inefficiency directly costs research time. Look at memory management (GPU and CPU), redundant computation, unnecessary I/O, inefficient data structures, and missed opportunities for batching or caching. Consider both per-record overhead and one-time costs that add up.

---

## PHASE 2 — VERIFY (Skeptical Verification Agent)

After ALL Phase 1 agents complete, collect their findings into a single list.

Spawn **1 verification agent** (subagent_type: `general-purpose`) that:

1. Takes the COMPLETE list of findings from all 5 agents
2. For EACH finding, uses the Read tool to read the actual source code at the cited file:line (and surrounding context)
3. Classifies each finding as:
   - **CONFIRMED** — The issue is real. Include the exact code snippet as proof.
   - **FALSE POSITIVE** — The code is actually correct. Explain why the original agent was wrong.
   - **NEEDS CONTEXT** — Cannot determine without runtime information. Explain what's missing.
4. Only **CONFIRMED** findings survive to Phase 3
5. Report the false positive rate for each agent

**CRITICAL**: The verification agent must ACTUALLY READ the source code. Do not trust Phase 1 descriptions at face value. Be genuinely skeptical — if the code looks correct, say so.

---

## PHASE 3 — SYNTHESIZE & INTERVIEW

### Step 1: Generate Report

Save a markdown report to `docs/code_review_report.md`:

```markdown
# Code Review Report
**Date**: {today}
**Scope**: {scope or "Full project"}
**Findings**: {confirmed_count} confirmed / {total_count} reviewed ({false_positive_pct}% false positive rate)

## CRITICAL
### C1: {title}
- **Location**: `file_path:line_number`
- **Root Cause**: {why this is wrong}
- **Impact**: {what breaks or could break}
- **Suggested Fix**: {concrete code change}
- **Test Impact**: {what tests to add/modify}

## HIGH
...
## MEDIUM
...
## LOW
...

## Agent Accuracy
| Agent | Findings | Confirmed | False Positive Rate |
|-------|----------|-----------|---------------------|
| Bug Hunter | N | N | X% |
| Test Coverage | N | N | X% |
| Error Handling | N | N | X% |
| Consistency | N | N | X% |
| Performance | N | N | X% |
```

### Step 2: Interactive Interview (unless --skip-interview)

Present findings ONE AT A TIME, starting from CRITICAL, then HIGH, MEDIUM, LOW.

For each finding, use AskUserQuestion:
- **Fix now** — Add to immediate implementation plan
- **Fix later** — Add to backlog
- **Skip** — Exclude (note reason)

### Step 3: Write Implementation Plan

After all decisions, write `docs/code_review_plan.md`:

```markdown
# Implementation Plan
**Generated from code review**: {date}

## Immediate Fixes
### Fix 1: {title}
- **File**: `path:line`
- **Change**: {description}
- **Test**: {what to verify}

## Backlog
### B1: {title}
- **File**: `path:line`
- **Change**: {description}
- **Priority**: {high/medium/low}

## Skipped
| Finding | Reason |
|---------|--------|
```
