# Claude Code Power-User Guide for SAE-Code-Correctness

*Compiled January 2026. Actionable improvements based on Boris Cherny's workflow, Anthropic official docs, and community best practices.*

---

## Current Setup Audit

| Component | Status | Gap |
|-----------|--------|-----|
| CLAUDE.md | **Strong** (~320 lines, well-structured) | Could modularize into `.claude/rules/` |
| Skills/Commands | **4 defined** (test-phase, hf-upload, perspectives, git-push-all) | Missing phase-runner, gpu-check, review skills |
| Hooks | **None configured** | No notifications, no auto-formatting, no safety guards |
| MCP Servers | **None** (only `mcp__ide__getDiagnostics`) | No GitHub MCP, no external integrations |
| Git Worktrees | **Single worktree** | Can't run parallel Claude sessions |
| Custom Subagents | **None** | No specialized agents for phases or review |
| Notifications | **None** | Can't leave Claude unattended on long tasks |
| `.claude/rules/` | **Not used** | All rules packed in CLAUDE.md |
| Permissions | **91 allow entries** in `settings.local.json` | Well configured, comprehensive |
| Model/Thinking | **Opus 4.5 + thinking always on** | Optimal (Boris Cherny recommends this) |

---

## Quick-Start Checklist (Top 5)

Do these first. Each takes minutes to set up and pays back immediately.

1. **Set up notification hooks** — get desktop + phone alerts when Claude finishes or needs input ([Section 1](#1-notifications--autonomous-operation))
2. **Add data-protection hook** — prevent accidental `rm -rf data/` during sessions ([Section 5](#5-hooks-configuration))
3. **Create worktrees for icml/main** — run parallel Claude sessions on different branches ([Section 2](#2-git-worktrees-for-parallel-sessions))
4. **Add auto-format hook** — ruff format on every Edit/Write ([Section 5](#5-hooks-configuration))
5. **Add `/run-phase` skill** — standardize phase execution with pre-checks ([Section 4](#4-expanded-skills))

---

## Improvement Areas

### 1. Notifications & Autonomous Operation

**Why it matters:** Phases run for hours. Without notifications you're either polling terminal tabs or missing when Claude needs input. Boris Cherny uses iTerm2 notifications to manage 10-15 concurrent sessions.

**Linux desktop + phone push notifications:**

Add to `.claude/settings.json`:

```json
{
  "permissions": {
    "allow": ["WebSearch", "WebFetch"]
  },
  "hooks": {
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' \"$(cat /dev/stdin | jq -r '.message // \"Needs attention\"')\" 2>/dev/null; curl -sf -d \"Claude: $(cat /dev/stdin | jq -r '.message // \"Needs attention\"')\" ntfy.sh/${NTFY_TOPIC:-claude-sae} >/dev/null 2>&1 || true"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' 'Task completed' 2>/dev/null; curl -sf -d 'Claude finished task' ntfy.sh/${NTFY_TOPIC:-claude-sae} >/dev/null 2>&1 || true"
          }
        ]
      }
    ]
  }
}
```

**Setup steps:**

```bash
# 1. Install notify-send (if not already available)
sudo apt-get install libnotify-bin

# 2. Set up ntfy.sh for phone notifications
# Install the ntfy app on your phone (Android/iOS)
# Subscribe to your topic in the app
echo 'export NTFY_TOPIC="sae-cc-alerts-$(whoami)"' >> ~/.bashrc
source ~/.bashrc

# 3. Test
notify-send 'Test' 'Desktop works'
curl -d "Phone works" ntfy.sh/$NTFY_TOPIC
```

**Hook events available:**

| Hook Event | When It Fires | Use Case |
|------------|---------------|----------|
| `Notification` | Claude sends any notification | Attention needed |
| `Stop` | Claude finishes responding | Task complete |
| `PermissionRequest` | Permission dialog appears | Approval needed |
| `SessionStart` | Session begins/resumes | Environment setup |
| `PreToolUse` | Before tool execution | Safety guards |
| `PostToolUse` | After tool succeeds | Auto-formatting |

---

### 2. Git Worktrees for Parallel Sessions

**Why it matters:** You have `main` and `icml` branches with different work. Worktrees let you run independent Claude Code sessions on each without conflicts.

**Setup:**

```bash
# From the main repo directory
cd /home/kriz.tahimic/sae-code-correctness

# Create worktrees for each active branch
git worktree add ../sae-cc-main main
git worktree add ../sae-cc-icml icml

# Each worktree gets its own Claude Code session
# Terminal 1:
cd ../sae-cc-main && claude

# Terminal 2:
cd ../sae-cc-icml && claude

# List worktrees
git worktree list

# Clean up when done
git worktree remove ../sae-cc-main
```

**Naming convention:** `../sae-cc-{branch}` keeps them adjacent and predictable.

**Important considerations:**
- Each worktree has independent file state — edits in one don't affect the other
- `.claude/` settings are shared (they're in the repo)
- `data/` is per-worktree — each gets its own output directories
- conda environment is shared (system-level, not per-directory)
- Both worktrees share Git history and remote connections

**When to use:**
- Running Phase 3.8 validation on `main` while writing new phase code on `icml`
- Testing HumanEval changes while MBPP run continues
- Code review on one branch while generation runs on another

**When NOT to use:**
- Tasks that need the same GPU (both would OOM)
- Tasks that write to the same output directory

**Boris Cherny's approach:** He uses separate git checkouts rather than worktrees, but worktrees are lighter-weight and share the object database.

---

### 3. Custom Subagents

**Why it matters:** Subagents are specialized Claude instances that the main Claude can delegate to. They get their own tools, model, and instructions.

**Location:** `.claude/agents/<name>.md`

#### Phase Runner Agent

File: `.claude/agents/phase-runner.md`

```yaml
---
name: phase-runner
description: Specialized agent for running and monitoring SAE phases. Use when executing phases, checking phase outputs, or debugging phase failures.
tools: Bash, Read, Glob, Grep
model: sonnet
---

You are a phase execution specialist for the SAE-code-correctness project.

## Environment
Always activate conda first:
```
source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc
```

## Before Running Any Phase
1. Check GPU memory: `nvidia-smi --query-gpu=memory.free --format=csv,noheader`
2. Verify dependencies exist using `discover_latest_phase_output()`
3. Test with `--start 0 --end 10` before full run
4. For parallel phases: use `--parallel 4`

## Monitoring
- Check output directory for checkpoint files
- Watch for OOM errors in output
- Verify pass rates are in expected ranges (25-35%)

## On Failure
- Report the exact error message
- Check if it's a GPU OOM (reduce batch size)
- Check if prior phase output is missing
- Never re-run without understanding the failure
```

#### GPU Monitor Agent

File: `.claude/agents/gpu-monitor.md`

```yaml
---
name: gpu-monitor
description: Check GPU usage, memory, and running processes. Use when troubleshooting OOM or checking if GPUs are free.
tools: Bash, Read
model: haiku
---

You monitor GPU status for this project. Run these commands and report findings:

1. `nvidia-smi` - Full GPU status
2. `nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv` - Summary table
3. `nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv` - Running processes

Report: which GPUs are free, which are in use, and estimated available memory.
```

#### Code Reviewer Agent

File: `.claude/agents/code-reviewer.md`

```yaml
---
name: code-reviewer
description: Reviews code changes for quality, correctness, and adherence to project conventions. Use before committing significant changes.
tools: Read, Glob, Grep, Bash
disallowedTools: Write, Edit
model: sonnet
---

You are a code reviewer for the SAE-code-correctness project.

## Review Checklist
1. **Terminology**: Uses `latent_idx` not `feature`, `latent_direction` not `feature_direction`
2. **Paths**: Uses `discover_latest_phase_output()` not hardcoded paths
3. **Tensors**: Uses `.safetensors` not `.npz`, uses `einops.rearrange` for complex reshapes
4. **No backward compatibility**: Clean breaks, no legacy fallbacks
5. **Color scheme**: Uses `COLOR_CORRECTION`, `COLOR_CORRUPTION`, `COLOR_PRESERVATION` constants
6. **Test outcome terms**: `baseline_passed`, `steered_correct`, `correction`, `corruption`, `preservation`
7. **Security**: No hardcoded secrets, no command injection risks

Run `git diff` to see changes, then review file by file.

Organize feedback as:
- **Critical** (must fix before commit)
- **Warning** (should fix)
- **Note** (consider for future)
```

**Frontmatter fields reference:**

| Field | Description |
|-------|-------------|
| `name` | Unique identifier (lowercase, hyphens) |
| `description` | When Claude should delegate to this agent |
| `tools` | Allowlist of tools (inherits all if omitted) |
| `disallowedTools` | Tools to deny |
| `model` | `sonnet`, `opus`, `haiku`, or `inherit` |
| `permissionMode` | `default`, `acceptEdits`, `dontAsk`, `plan` |

**Key behaviors:**
- Subagents cannot spawn other subagents (no nesting)
- Press `Ctrl+B` to background a running subagent
- Claude can resume a subagent with full conversation history

---

### 4. Expanded Skills

**Current skills:** test-phase, hf-upload, perspectives, git-push-all

**Recommended additions:**

#### `/run-phase` — Phase Execution with Pre-Checks

File: `.claude/commands/run-phase.md`

```yaml
description: Run a phase with standard pre-checks (GPU, dependencies, conda)
argument-hint: "<phase_number> [--parallel N] [--start S --end E] [--direction-source SOURCE]"
allowed-tools: Bash(*), Read(*), Grep(*), Glob(*)
```

Execute a phase with safety checks:

1. **Activate environment:**
   ```
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc
   ```

2. **Pre-flight checks:**
   - Run `nvidia-smi --query-gpu=memory.free --format=csv,noheader` — ensure at least 20GB free per GPU
   - Verify the phase's input dependencies exist using auto-discovery
   - Check disk space: `df -h /home/kriz.tahimic/` — ensure at least 10GB free

3. **Execute the phase:**
   ```
   python3 run.py phase $ARGUMENTS
   ```

4. **Post-run validation:**
   - Check exit code
   - Verify output files were created in the expected directory
   - Report summary of what was produced

If any pre-flight check fails, report the issue and do NOT run the phase.

---

#### `/check-gpu` — Quick GPU Status

File: `.claude/commands/check-gpu.md`

```yaml
description: Quick GPU status check showing memory and running processes
allowed-tools: Bash(nvidia-smi*)
```

Run these commands and present a clean summary:

```bash
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=pid,gpu_uuid,used_memory --format=csv,noheader 2>/dev/null || echo "No processes running"
```

Format as a table showing: GPU index, memory used/total, utilization %, and any running processes.

---

#### `/review` — Pre-Commit Code Review

File: `.claude/commands/review.md`

```yaml
description: Review staged/unstaged changes against project conventions before committing
allowed-tools: Bash(git*), Read(*), Grep(*), Glob(*)
```

Review all uncommitted changes for project convention compliance:

1. Run `git diff` and `git diff --cached` to see all changes
2. Check each modified file against these rules:
   - Uses `latent_idx` not `feature` or `feature_index`
   - Uses `discover_latest_phase_output()` not hardcoded paths
   - Uses `.safetensors` not `.npz` for new tensor storage
   - Uses `einops.rearrange` for complex reshapes
   - No backward compatibility fallbacks (clean breaks only)
   - Color constants from `common/config.py`
3. Report any violations organized by severity

---

#### `/resume-phase` — Resume Interrupted Phase

File: `.claude/commands/resume-phase.md`

```yaml
description: Detect and resume an interrupted phase from its last checkpoint
argument-hint: "<phase_number> [--parallel N]"
allowed-tools: Bash(*), Read(*), Grep(*), Glob(*)
```

Detect and resume an interrupted phase:

1. **Activate environment:**
   ```
   source ~/miniconda3/etc/profile.d/conda.sh && conda activate sae_cc
   ```

2. **Find checkpoints:** Look in the phase's output directory for checkpoint files or partial outputs

3. **Determine resume point:** Report what was completed and what remains

4. **Re-run the same command** — checkpointing is automatic, the phase will skip completed work:
   ```
   python3 run.py phase $ARGUMENTS
   ```

5. **Monitor:** Watch for the "Resuming from checkpoint" message in output

---

### 5. Hooks Configuration

**Why it matters:** Hooks automate repetitive actions and add safety guardrails. No hooks are currently configured despite the project documenting the pattern.

**Recommended hooks to add to `.claude/settings.json`:**

```json
{
  "permissions": {
    "allow": ["WebSearch", "WebFetch"]
  },
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "hooks": [
          {
            "type": "command",
            "command": "cd \"$CLAUDE_PROJECT_DIR\" && ruff format --quiet $(echo '$TOOL_INPUT' | jq -r '.file_path // empty') 2>/dev/null || true",
            "timeout": 10
          }
        ]
      }
    ],
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "echo '$TOOL_INPUT' | jq -r '.command // empty' | grep -qE 'rm\\s+(-rf?|--recursive).*data/' && echo 'BLOCKED: Cannot rm -rf data/ directory' >&2 && exit 2 || true",
            "timeout": 5
          }
        ]
      }
    ],
    "Notification": [
      {
        "matcher": "",
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' \"$(cat /dev/stdin | jq -r '.message // \"Needs attention\"')\" 2>/dev/null || true"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "notify-send 'Claude Code' 'Task completed' 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}
```

**What each hook does:**

| Hook | Event | Purpose |
|------|-------|---------|
| PostToolUse `Write\|Edit` | After file edits | Auto-format Python with ruff |
| PreToolUse `Bash` | Before bash commands | Block `rm -rf data/` (exit code 2 = block) |
| Notification | Any notification | Desktop alert via notify-send |
| Stop | Claude finishes | Desktop alert that task is done |

**Hook exit codes:**

| Code | Behavior |
|------|----------|
| `0` | Success, continue |
| `2` | **Block the action** — stderr fed back to Claude |
| Other | Non-blocking error, continue |

**Advanced: Hook input/output format**

Hooks receive JSON via stdin with session context:

```json
{
  "session_id": "abc123",
  "hook_event_name": "PreToolUse",
  "tool_name": "Bash",
  "tool_input": {
    "command": "rm -rf data/phase1_0",
    "description": "Delete phase output"
  }
}
```

PreToolUse hooks can return JSON to auto-approve, deny, or modify the input:

```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "Cannot delete data directories"
  }
}
```

---

### 6. Modular Rules (`.claude/rules/`)

**Why it matters:** CLAUDE.md is ~320 lines. Some rules are only relevant when working on specific directories. Modular rules let you scope instructions to file paths.

**Proposed structure:**

```
.claude/rules/
├── terminology.md          # Always loaded: SAE terminology standards
├── phase-conventions.md    # Always loaded: phase development patterns
├── tensor-patterns.md      # Path-scoped: *.py files
└── steering.md             # Path-scoped: phase4_*/**, phase5_*/**, phase8_*/**
```

**Unconditional rule (no frontmatter, always loaded):**

File: `.claude/rules/terminology.md`

```markdown
# SAE Terminology

Use these terms exactly:
- `latent_idx` (int) — NOT `feature`, `feature_index`, `latent_index`
- `latent_direction` (Tensor) — NOT `feature_direction`
- `latent_activation` (float) — NOT `feature_activation`
- `latent_activations` (Tensor) — NOT `feature_activations`
- Say "predicting" not "preferring" or "detecting"

Test outcome terms:
- `baseline_passed` — did unmodified generation pass?
- `steered_correct` — is steered output correct?
- `correction` = baseline failed AND steered correct
- `corruption` = baseline passed AND steered failed
- `preservation` = baseline passed AND steered correct
```

**Path-scoped rule (loaded only for matching files):**

File: `.claude/rules/steering.md`

```yaml
---
paths:
  - "phase4_*/**"
  - "phase5_*/**"
  - "phase8_*/**"
  - "common/steering_setup.py"
---

# Steering Phase Rules

Steering formula:
  steered = original + coefficient * direction

Direction sources:
- SAE (Phase 2.5): separation score metric, used for steering phases
- Probe LogReg (Phase 2.6): supervised, best for detection (AUROC/F1)
- Probe Mass-Mean (Phase 2.6): supervised, best for steering correction

Phases 8.2/8.3 use BOTH probes internally — logreg for threshold, mass_mean for steering.

Coefficient search: coarse-to-fine (Phase 4.5) → golden section (Phase 4.6).
```

**Migration approach:**
1. Extract focused rules from CLAUDE.md sections 5 (Architecture) and 6 (Standards)
2. Keep CLAUDE.md as the overview and reference
3. Rules in `.claude/rules/` supplement, not replace, CLAUDE.md

**Priority order (Claude loads in this order):**
1. Managed policy (highest)
2. Project memory (CLAUDE.md / `.claude/CLAUDE.md`)
3. Project rules (`.claude/rules/*.md`)
4. User memory (`~/.claude/CLAUDE.md`)
5. Local memory (`CLAUDE.local.md`)

---

### 7. MCP Servers

**Why it matters:** MCP (Model Context Protocol) servers give Claude access to external tools. Currently only IDE diagnostics is configured.

**Configuration location:** `.mcp.json` (project root, check into git) or `~/.claude.json` (user-level).

#### GitHub MCP Server

Useful for PR management, issue tracking, and code review without leaving Claude.

File: `.mcp.json`

```json
{
  "mcpServers": {
    "github": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": {
        "GITHUB_PERSONAL_ACCESS_TOKEN": "${GITHUB_TOKEN}"
      }
    }
  }
}
```

Setup:

```bash
# Ensure GITHUB_TOKEN is set
echo 'export GITHUB_TOKEN="ghp_..."' >> ~/.bashrc
source ~/.bashrc

# Or use gh auth token
export GITHUB_TOKEN=$(gh auth token)
```

**Note:** `gh` CLI already works well from Bash. MCP adds tool-native integration but isn't strictly necessary if you're comfortable with `gh pr`, `gh issue`, etc.

#### Memory MCP Server

Persistent memory across sessions — useful for tracking experiment results.

```json
{
  "mcpServers": {
    "memory": {
      "type": "stdio",
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    }
  }
}
```

**Recommendation:** Low priority. The project's checkpoint system and CLAUDE.md already handle most memory needs. Consider only if you find yourself repeatedly re-explaining experiment context across sessions.

---

### 8. CLAUDE.md Refinements

**Current state:** Well-structured at ~320 lines. The main improvements are about workflow, not content.

#### Use `/memory` for Iterative Improvement

When Claude makes a mistake, instead of manually editing CLAUDE.md:

```
/memory "Never use hardcoded paths like data/phase3_5_humaneval. Always use discover_latest_phase_output()."
```

This appends to CLAUDE.md automatically. Review periodically and clean up.

#### Use `@` Imports for Supporting Docs

Reference other files from CLAUDE.md without inlining their content:

```markdown
For steering implementation details, see @common/steering_setup.py
For parallel runner architecture, see @common/parallel_runner.py
```

Claude will read these files when relevant context is needed.

#### Prune Rules Claude Follows Naturally

Some rules in CLAUDE.md may be unnecessary because Claude already follows them (e.g., basic Python style, not using deprecated syntax). Monitor which rules are actually violated and remove ones that never trigger. Keep CLAUDE.md focused on project-specific conventions that Claude wouldn't infer on its own.

---

## Implementation Priority

| # | Improvement | Priority | Effort | Impact |
|---|-------------|----------|--------|--------|
| 1 | Notification hooks (desktop + phone) | **HIGH** | 10 min | Leave Claude unattended during long phases |
| 2 | Data-protection hook (`rm -rf data/`) | **HIGH** | 5 min | Prevent catastrophic data loss |
| 3 | Git worktrees for icml/main | **HIGH** | 5 min | Parallel sessions on different branches |
| 4 | Auto-format hook (ruff) | **MEDIUM** | 5 min | Consistent formatting without manual steps |
| 5 | `/run-phase` skill | **MEDIUM** | 10 min | Standardized phase execution |
| 6 | `/check-gpu` skill | **MEDIUM** | 5 min | Quick GPU status without remembering nvidia-smi flags |
| 7 | Custom subagents (phase-runner, gpu-monitor) | **MEDIUM** | 15 min | Specialized delegation |
| 8 | `/review` skill | **MEDIUM** | 5 min | Convention compliance before commits |
| 9 | Modular rules (`.claude/rules/`) | **LOW** | 20 min | Path-scoped instructions |
| 10 | GitHub MCP server | **LOW** | 10 min | Native GitHub integration |
| 11 | CLAUDE.md refinements | **LOW** | Ongoing | Iterative improvement |

---

## Portability: New Machine / New Project Setup

### What travels with `git clone` (no setup needed)

Everything in `.claude/` is checked into git and works immediately:

| File | Contains |
|------|----------|
| `.claude/settings.json` | Hooks (notifications, data protection, auto-format) |
| `.claude/commands/*.md` | Skills (test-phase, hf-upload, check-gpu, etc.) |
| `.claude/agents/*.md` | Custom subagents |
| `.claude/rules/*.md` | Modular rules |
| `CLAUDE.md` | Project instructions |
| `.mcp.json` | MCP server definitions |

### What needs per-machine setup

| Item | What to do | One-liner |
|------|-----------|-----------|
| **ntfy.sh topic** | Set env var + subscribe on phone | `echo 'export NTFY_TOPIC="your-topic"' >> ~/.bashrc` |
| **ruff** | Install in your Python env | `pip install ruff` |
| **jq** | Install (used by hooks) | `sudo apt-get install jq` |
| **curl** | Install (used by notification hooks) | Usually pre-installed |
| **Permissions** | Re-approve on first use | Claude Code prompts automatically |
| **conda env** | Recreate from requirements | `conda create -n sae_cc python=3.11 && pip install -r requirements.txt` |

### Reusing this workflow in a new project

To copy the Claude Code setup to a different repo:

```bash
# From your new project root
mkdir -p .claude/commands .claude/agents .claude/rules

# Copy the portable config files
cp /path/to/sae-code-correctness/.claude/settings.json .claude/settings.json
cp /path/to/sae-code-correctness/.claude/commands/*.md .claude/commands/

# Edit settings.json to:
# 1. Update the ntfy.sh topic (or keep using NTFY_TOPIC env var)
# 2. Adjust the PreToolUse data-protection hook for your project's dirs
# 3. Keep PostToolUse (ruff) and notification hooks as-is

# Write a new CLAUDE.md for the new project
# (don't copy the SAE-specific one)
```

### Design decisions for portability

The hooks in `.claude/settings.json` are designed to be portable:

- **ruff**: Uses bare `ruff` (PATH lookup), not a hardcoded conda path. Fails silently if not installed.
- **ntfy.sh**: Uses `${NTFY_TOPIC:-sae-cc-kriz-8eb523ea}` — env var with fallback. Change the env var per machine.
- **notify-send**: Fails silently with `2>/dev/null` on machines without a display (e.g., remote servers).
- **Terminal bell**: `printf '\a' > /dev/tty` works over SSH if the client supports it. Fails silently otherwise.
- **Data protection**: Uses regex pattern matching, not hardcoded paths. Adjust the `data/` pattern for your project.
- **All hooks end with `; true`** or `|| true` so they never block Claude on transient failures.

---

## Reference: Hook Events Complete List

| Event | Matcher Options | Use Case |
|-------|----------------|----------|
| `SessionStart` | `startup`, `resume`, `clear`, `compact` | Environment setup |
| `UserPromptSubmit` | — | Prompt validation |
| `PreToolUse` | Tool names: `Bash`, `Edit`, `Write`, `Read`, etc. | Safety guards, auto-approve |
| `PostToolUse` | Tool names | Auto-formatting, logging |
| `PostToolUseFailure` | Tool names | Error handling |
| `PermissionRequest` | — | Attention alerts |
| `SubagentStart` | — | Subagent lifecycle |
| `SubagentStop` | — | Subagent lifecycle |
| `Stop` | — | Task completion notification |
| `PreCompact` | `manual`, `auto` | Context management |
| `Notification` | `permission_prompt`, `idle_prompt` | Alerts |
| `SessionEnd` | — | Cleanup |
| `Setup` | — | Initialization (`--init`, `--maintenance`) |

## Reference: Skills vs Commands

| Feature | Commands (`.claude/commands/`) | Skills (`.claude/skills/`) |
|---------|------|--------|
| Format | Single `.md` file | Directory with `SKILL.md` + optional files |
| Auto-invocation by Claude | No | Yes (unless `disable-model-invocation: true`) |
| Supporting files | No | Yes (scripts, templates, examples) |
| Hooks | No | Yes (scoped to skill lifecycle) |
| Subagent execution | No | Yes (`context: fork`) |
| Dynamic context | Yes (`!`cmd``) | Yes |

**Recommendation:** Continue using `.claude/commands/` for simple skills. Migrate to `.claude/skills/` only when you need supporting files, hooks, or subagent execution.

## Reference: Subagent Frontmatter

```yaml
---
name: agent-name                    # Required: unique ID
description: When to use this agent # Required: Claude uses this for delegation
tools: Bash, Read, Grep             # Optional: tool allowlist
disallowedTools: Write, Edit        # Optional: tool denylist
model: sonnet                       # Optional: sonnet, opus, haiku, inherit
permissionMode: default             # Optional: default, acceptEdits, dontAsk, plan
skills:                             # Optional: skills to preload
  - skill-name
hooks:                              # Optional: lifecycle hooks scoped to agent
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate.sh"
---

Agent instructions go here (markdown body).
```

---

## Sources

1. **Boris Cherny's Workflow** (creator of Claude Code)
   - [VentureBeat coverage](https://venturebeat.com/technology/the-creator-of-claude-code-just-revealed-his-workflow-and-developers-are)
   - [InfoQ summary](https://www.infoq.com/news/2026/01/claude-code-creator-workflow/)

2. **Official Claude Code Docs**
   - [Hooks](https://code.claude.com/docs/en/hooks)
   - [Skills](https://code.claude.com/docs/en/skills)
   - [Subagents](https://code.claude.com/docs/en/sub-agents)
   - [Memory & Rules](https://code.claude.com/docs/en/memory)
   - [MCP Servers](https://code.claude.com/docs/en/mcp)
   - [Common Workflows](https://code.claude.com/docs/en/common-workflows)

3. **Git Worktrees + Claude Code**
   - [incident.io blog](https://incident.io/blog/shipping-faster-with-claude-code-and-git-worktrees)
   - [GitButler parallel sessions](https://blog.gitbutler.com/parallel-claude-code)

4. **Notifications**
   - [Andrew Ford — ntfy.sh guide](https://andrewford.co.nz/articles/claude-code-instant-notifications-ntfy/)
   - [Boris Buliga — notifications](https://www.d12frosted.io/posts/2026-01-05-claude-code-notifications)

5. **Community Resources**
   - [Awesome Claude Code Subagents](https://github.com/VoltAgent/awesome-claude-code-subagents)
   - [Claude Code Orchestrator](https://news.ycombinator.com/item?id=46578028)
