---
description: Stage all changes, commit, and push (optional commit message)
argument-hint: "[optional commit message]"
allowed-tools: Bash(git add:*), Bash(git commit:*), Bash(git push:*), Bash(git status:*), Bash(git diff:*)
---

Stage all changes, commit, and push to the current branch.

If a commit message is provided below, use it. Otherwise, analyze the changes and compose a concise, descriptive commit message.

User-provided message: $ARGUMENTS
