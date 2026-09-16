---
name: adapt-claude-to-cursor
description: >-
  Convert Claude Code configuration (CLAUDE.md and .claude/skills/) to Cursor
  format (.cursor/rules/ and .cursor/skills/). Automatically adapts any
  repository's CLAUDE.md into .mdc rule files and .claude/skills/ SKILL.md
  files into Cursor-compatible SKILL.md format. Use when the user wants to
  migrate, convert, or adapt Claude configuration to Cursor, or mentions
  CLAUDE.md, .claude/skills, or Claude-to-Cursor migration.
---

# Adapt Claude Configuration to Cursor

Convert a repository's Claude Code configuration into Cursor-native format.

## Overview

Claude Code uses `CLAUDE.md` (project guide) and `.claude/skills/` (task skills).
Cursor uses `.cursor/rules/*.mdc` (rules) and `.cursor/skills/*/SKILL.md` (skills).
This skill automates the conversion.

## Format Mapping

| Claude | Cursor | Notes |
|--------|--------|-------|
| `CLAUDE.md` (root) | `.cursor/rules/*.mdc` | Split into focused rule files |
| `.claude/skills/*/SKILL.md` | `.cursor/skills/*/SKILL.md` | Adapt frontmatter |
| `AGENTS.md` (if exists) | `.cursor/rules/*.mdc` | Same as CLAUDE.md |

## Step 1: Discover Source Files

Scan the target repository for Claude configuration:

```bash
# Find CLAUDE.md files (root and subdirectories)
find <REPO_ROOT> -maxdepth 2 -name "CLAUDE.md" -type f 2>/dev/null

# Find .claude/skills
find <REPO_ROOT>/.claude/skills -name "SKILL.md" -type f 2>/dev/null

# Find AGENTS.md files
find <REPO_ROOT> -maxdepth 2 -name "AGENTS.md" -type f 2>/dev/null
```

If no Claude configuration is found, inform the user and stop.

## Step 2: Create Target Directories

```bash
mkdir -p <REPO_ROOT>/.cursor/rules
mkdir -p <REPO_ROOT>/.cursor/skills
```

## Step 3: Convert CLAUDE.md → .cursor/rules/*.mdc

### 3.1 Read and Analyze CLAUDE.md

Read the full content of `CLAUDE.md`. Identify logical sections by top-level
headings (`##`). Group related sections into rule files by theme.

### 3.2 Splitting Strategy

A single `CLAUDE.md` should be split into **1-3 focused `.mdc` files** based on content:

| Content Theme | Rule File Name | Trigger |
|---------------|----------------|---------|
| Project overview, repo layout, build, test, code style | `<project>-project.mdc` | `alwaysApply: true` |
| Language/framework-specific patterns, API usage | `<project>-patterns.mdc` | `globs: **/*.py` (or relevant extension) |
| Architecture, design decisions | `<project>-architecture.mdc` | `alwaysApply: true` |

If the CLAUDE.md is short (< 80 lines), keep it as a single `.mdc` file.

### 3.3 .mdc File Format

Each `.mdc` file must have this structure:

```markdown
---
description: One-line description of what this rule covers
globs: **/*.py          # Optional: only include if rule is file-type specific
alwaysApply: true       # Set true for project-wide rules, false for file-specific
---

# Rule Title

Rule content here (converted from CLAUDE.md sections)...
```

### 3.4 Content Adaptation Rules

When converting CLAUDE.md content to `.mdc`:

1. **Keep it concise** — Cursor rules should ideally be under 50 lines each.
   Move detailed reference material to skills instead.
2. **Remove Claude-specific instructions** — Strip any references to Claude's
   slash commands (`/command`), tool names (`Bash`, `Read`, `Edit`), or
   Claude-specific behaviors.
3. **Remove `Usage:` lines** — Claude skills use `Usage: /skill-name` which
   doesn't apply in Cursor.
4. **Preserve code examples** — Keep all code snippets, they're valuable context.
5. **Preserve domain knowledge** — Keep all project-specific facts, conventions,
   and gotchas.

## Step 4: Convert .claude/skills/ → .cursor/skills/

### 4.1 For Each Skill Directory

For each `SKILL.md` found in `.claude/skills/<skill-name>/`:

1. Create the target directory: `mkdir -p .cursor/skills/<skill-name>/`
2. Read the source `SKILL.md`
3. Transform the frontmatter (see 4.2)
4. Transform the body (see 4.3)
5. Write the result to `.cursor/skills/<skill-name>/SKILL.md`
6. Copy any companion files (reference.md, scripts/) if they exist

### 4.2 Frontmatter Transformation

**Claude SKILL.md frontmatter** may contain these fields:

```yaml
---
name: skill-name
description: What the skill does
tools: Read,Edit,Bash,Grep,Glob,Agent    # Claude-specific
user_invocable: true                      # Claude-specific
---
```

**Cursor SKILL.md frontmatter** requires only:

```yaml
---
name: skill-name
description: What the skill does and when to use it
---
```

Transformation rules:

| Claude Field | Action |
|--------------|--------|
| `name` | **Keep** — copy as-is (must be lowercase, hyphens, max 64 chars) |
| `description` | **Keep and enhance** — if it doesn't include "Use when...", append trigger scenarios |
| `tools` | **Remove** — Cursor doesn't use this field |
| `user_invocable` | **Remove** — Cursor doesn't use this field |
| Any other fields | **Remove** — only `name` and `description` are recognized |

### 4.3 Description Enhancement

If the Claude description doesn't include trigger terms, enhance it:

**Before** (Claude):
```yaml
description: Optimize LDS access patterns in FlyDSL GPU kernels.
```

**After** (Cursor):
```yaml
description: >-
  Optimize LDS access patterns in FlyDSL GPU kernels. Diagnose bank
  conflicts and high lgkmcnt stalls from ATT traces. Use when trace
  analysis shows ds_read/ds_write/lgkmcnt as a bottleneck.
```

Guidelines for description:
- Write in **third person**
- Include both **WHAT** (capabilities) and **WHEN** (trigger scenarios)
- Include specific keywords that the user might mention
- Max 1024 characters

### 4.4 Body Transformation

When converting the skill body content:

1. **Remove `Usage: /skill-name` lines** — Cursor discovers skills automatically
   via description matching, not slash commands.
2. **Remove `tools:` references in body** — Lines like "This skill uses Read,
   Edit, Bash tools" are Claude-specific.
3. **Keep all technical content** — Domain knowledge, code examples, workflows,
   checklists are all valuable.
4. **Keep under 500 lines** — If the source is longer, move detailed reference
   material to companion files (reference.md, examples.md) with links from
   SKILL.md.
5. **Preserve companion files** — If `.claude/skills/<name>/` contains files
   besides SKILL.md (scripts, references), copy them to `.cursor/skills/<name>/`.

## Step 5: Handle AGENTS.md (if present)

`AGENTS.md` files serve the same role as `CLAUDE.md` but may exist in
subdirectories. Convert them using the same rules as Step 3, with the
rule's `globs:` set to match the subdirectory's file patterns.

For example, `frontend/AGENTS.md` becomes:

```markdown
---
description: Frontend development conventions
globs: frontend/**
alwaysApply: false
---
```

## Step 6: Verify and Report

After conversion, report a summary:

```
Conversion complete:

Rules created (.cursor/rules/):
  - <project>-project.mdc (alwaysApply: true, XX lines)
  - <project>-patterns.mdc (globs: **/*.py, XX lines)

Skills converted (.cursor/skills/):
  - skill-name-1/SKILL.md (XX lines)
  - skill-name-2/SKILL.md (XX lines)
  ...

Source files (NOT modified):
  - CLAUDE.md
  - .claude/skills/skill-name-1/SKILL.md
  - .claude/skills/skill-name-2/SKILL.md
  ...
```

**Important**: Never modify or delete the original Claude files. The two
configurations can coexist — Claude reads `.claude/` and Cursor reads `.cursor/`.

## Common Pitfalls

1. **Don't create rules in `~/.cursor/skills-cursor/`** — that's reserved for
   Cursor's internal built-in skills.
2. **Don't exceed 500 lines per SKILL.md** — use progressive disclosure
   (companion files) for long skills.
3. **Don't mix formats** — `.mdc` files go in `.cursor/rules/`,
   `SKILL.md` files go in `.cursor/skills/<name>/`.
4. **Don't forget `alwaysApply` or `globs`** — every `.mdc` file must specify
   one of these so Cursor knows when to apply it.
5. **Don't include Claude tool names** — `tools: Read,Edit,Bash` is meaningless
   in Cursor.
6. **Rule file names must end in `.mdc`** — not `.md`.

## Quick Reference: Format Comparison

| Aspect | Claude | Cursor |
|--------|--------|--------|
| Project guide | `CLAUDE.md` in root | `.cursor/rules/*.mdc` |
| Subdirectory guide | `AGENTS.md` | `.cursor/rules/*.mdc` with `globs:` |
| Skills location | `.claude/skills/<name>/SKILL.md` | `.cursor/skills/<name>/SKILL.md` |
| Skill frontmatter | `name`, `description`, `tools`, `user_invocable` | `name`, `description` only |
| Skill discovery | `/skill-name` slash command | Auto-matched via `description` keywords |
| Rule trigger | Always loaded | `alwaysApply: true` or `globs: pattern` |
| Max lengths | No hard limit | Rules: ~50 lines ideal; Skills: <500 lines |
