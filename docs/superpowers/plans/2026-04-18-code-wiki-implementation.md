# Code Wiki Skill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a reusable skill for building and maintaining code knowledge bases from source repositories.

**Architecture:** A single SKILL.md file that defines Build/Query/Lint operations. Build uses hybrid extraction (auto file/function parsing + AI interpretation). Query always combines Wiki context with source reading. Lint checks for outdated/orphaned pages.

**Tech Stack:** Skill format (Markdown + YAML frontmatter), glob/grep for auto-extraction

---

## Task 1: Create Skill Directory Structure

**Files:**
- Create: `~/.claude/skills/code-wiki/`

- [ ] **Step 1: Create directory**

```bash
mkdir -p ~/.claude/skills/code-wiki
```

- [ ] **Step 2: Verify directory exists**

```bash
ls -la ~/.claude/skills/code-wiki
```

Expected: Empty directory

---

## Task 2: Write SKILL.md Frontmatter and Overview

**Files:**
- Create: `~/.claude/skills/code-wiki/SKILL.md`

- [ ] **Step 1: Write frontmatter**

```markdown
---
name: code-wiki
description: Use when building or querying a code knowledge base from a repository. Triggers: "理解仓库", "代码知识库", "build code wiki", "查代码", "lint wiki". Keywords: wiki, knowledge base, codebase understanding, documentation generation.
---
```

- [ ] **Step 2: Write Overview section**

```markdown
# Code Wiki

A skill for building and maintaining a code knowledge base (Wiki) from source code repositories. Combines automatic structure extraction with AI-driven design interpretation.

## Three-Layer Architecture

| Layer | Location | Content | Management |
|-------|----------|---------|------------|
| **Sources** | `code-wiki/sources/` | Code snapshots (versioned) | Automated |
| **Wiki** | `code-wiki/` | Module, concept, design pages | AI maintained |
| **Index** | `code-wiki/index.md` | Content index + search entry | AI updated |
```

---

## Task 3: Write When to Use Section

**Files:**
- Modify: `~/.claude/skills/code-wiki/SKILL.md`

- [ ] **Step 1: Add When to Use section after Overview**

```markdown
## When to Use

**Build** — Initialize or update the wiki:
- User says: "理解这个仓库" / "build code wiki" / "创建代码知识库"
- New code added, wiki needs refresh

**Query** — Answer code questions:
- User asks about code structure, design, implementation
- User wants to understand a module

**Lint** — Check wiki health:
- User says: "检查知识库" / "维护 wiki" / "lint wiki"

### Directory Structure

```
code-wiki/
├── index.md              # Content index
├── log.md                # Operation log
├── modules/              # Module pages
│   ├── {module-name}.md
├── concepts/             # Concept pages
│   ├── {concept-name}.md
├── design/               # Design decision pages
│   ├── {decision-name}.md
└── sources/              # Code snapshots (optional)
```
```

---

## Task 4: Write Build Operation

**Files:**
- Modify: `~/.claude/skills/code-wiki/SKILL.md`

- [ ] **Step 1: Add Build section**

```markdown
## Build Operation

When user says "理解仓库", "build code wiki", or "创建代码知识库":

### Step 1: Create Directory Structure

```bash
mkdir -p code-wiki/{modules,concepts,design,sources}
touch code-wiki/index.md code-wiki/log.md
```

### Step 2: Auto-Extract Code Structure

Extract using glob and grep:
- File tree: `find . -type f -name "*.py" | head -50`
- Function signatures: `grep -r "def \|class " --include="*.py"`
- Import relationships: `grep -r "^import \|^from " --include="*.py"`

### Step 3: AI Analysis

Analyze key modules for:
- Module purpose and responsibilities
- Key classes and their relationships
- Design patterns used
- Important algorithms

### Step 4: Generate Wiki Pages

Create pages per module/concept/design:

**Module page template:**
```markdown
# {Module Name}

## Purpose
{Brief description of what this module does}

## Key Classes/Functions
- `{ClassName}`: {purpose}
- `{FunctionName}`: {purpose}

## Relationships
- Depends on: {other modules}
- Used by: {other modules}

## Notes
{Design decisions, important patterns}
```

### Step 5: Update Index

Add new pages to `code-wiki/index.md`:
```markdown
## Modules
- [agent](modules/agent.md) - LLM providers, prompts, debate

## Concepts
- [expression-tree](concepts/expression-tree.md) - Expression tree representation
```

### Step 6: Log Operation

Append to `code-wiki/log.md`:
```markdown
## [YYYY-MM-DD] build | Initial wiki build
- Created N module pages
- Created M concept pages
- Key modules: agent, core, operators
```
```

---

## Task 5: Write Query Operation

**Files:**
- Modify: `~/.claude/skills/code-wiki/SKILL.md`

- [ ] **Step 1: Add Query section**

```markdown
## Query Operation

When user asks about code structure, design, or implementation:

### Step 1: Check Wiki First

1. Read `code-wiki/index.md` to locate relevant pages
2. Read the specific module/concept pages
3. Note design context and relationships

### Step 2: Read Source with Context

1. Read source files with Wiki context in mind
2. Cross-reference what Wiki says vs what code actually does
3. Note any discrepancies

### Step 3: Synthesize Answer

1. Answer based on combined Wiki + source knowledge
2. Cite specific file locations and line numbers
3. If insights not in Wiki, offer to add them

### Output Format

Prefer concise answers with code references:
```markdown
The `agent/factor_generator.py` handles factor generation.
Key class: `FactorGenerator` (line 45) - generates factors via LLM.

Related: See [expression-tree](../concepts/expression-tree.md) for how
factors are represented as expression trees.
```
```

---

## Task 6: Write Lint Operation

**Files:**
- Modify: `~/.claude/skills/code-wiki/SKILL.md`

- [ ] **Step 1: Add Lint section**

```markdown
## Lint Operation

When user says "检查知识库", "维护 wiki", or "lint wiki":

### Step 1: Check for Outdated Content

Compare `code-wiki/` pages against actual code:
- Module page mentions a file that no longer exists
- Function/class referenced was renamed or deleted
- Import relationships are broken

### Step 2: Check for Orphaned Pages

- Pages with no incoming links from other wiki pages
- Pages not linked from `index.md`

### Step 3: Check for Knowledge Gaps

- Important modules without wiki pages
- Core concepts not documented

### Step 4: Report Issues

Present findings:
```markdown
## Wiki Health Report

### Outdated Content
- `modules/core-parser.md`: references `parser.py` which was split into `parser.py` and `lexer.py`

### Orphaned Pages
- `concepts/debate.md` - not linked from any page

### Knowledge Gaps
- `operators/gpu_backend.py` - no wiki page
- Cross-sectional operators not documented
```
```

---

## Task 7: Write Quick Reference and Common Mistakes

**Files:**
- Modify: `~/.claude/skills/code-wiki/SKILL.md`

- [ ] **Step 1: Add Quick Reference section**

```markdown
## Quick Reference

### Build Trigger Phrases
- "理解这个仓库" / "understand this codebase"
- "创建代码知识库" / "build code wiki"
- "初始化 wiki"

### Query Trigger Phrases
- "这个模块怎么工作" / "how does this module work"
- "解释下代码" / "explain the code"
- "这个函数在哪" / "where is this function"

### Lint Trigger Phrases
- "检查知识库" / "check wiki"
- "维护 wiki" / "maintain wiki"
- "lint wiki"

### Directory Convention
Always create wiki in `code-wiki/` subdirectory of the repository root.
```

- [ ] **Step 2: Add Common Mistakes section**

```markdown
## Common Mistakes

**Wiki as once-only** — Don't build once and forget. Update wiki when code changes significantly.

**Copying code instead of explaining** — Wiki pages should explain WHY, not just WHAT. Show the intent and design.

**Missing cross-references** — Every page should link to related pages. Isolated pages become stale.

**Ignoring Query context** — Always read Wiki before diving into source. The Wiki provides context that makes source reading more efficient.
```

---

## Task 8: Verify Skill File

**Files:**
- Modify: `~/.claude/skills/code-wiki/SKILL.md`

- [ ] **Step 1: Verify YAML frontmatter is valid**

Frontmatter must have `name` and `description` fields, max 1024 chars total.

- [ ] **Step 2: Verify file exists in correct location**

```bash
ls -la ~/.claude/skills/code-wiki/SKILL.md
```

Expected: File exists with content

- [ ] **Step 3: Check skill can be loaded**

The skill file should be valid Markdown with proper structure.

---

## Task 9: Commit Skill to Git (Optional)

**Files:**
- Modify: `~/.claude/skills/code-wiki/SKILL.md`

- [ ] **Step 1: Initialize git in skills directory (if not already)**

```bash
cd ~/.claude/skills/code-wiki && git init
```

- [ ] **Step 2: Commit the skill**

```bash
git add SKILL.md && git commit -m "feat: add code-wiki skill for codebase knowledge management"
```

---

## Self-Review Checklist

After writing the plan, verify:

1. **Spec coverage:** Does each requirement from the design spec have a corresponding task?
   - Build operation ✓
   - Query operation ✓
   - Lint operation ✓
   - Directory structure ✓
   - Three-layer architecture ✓

2. **No placeholders:** Check for "TBD", "TODO", "fill in later" — none should exist

3. **File paths:** All paths are concrete (e.g., `~/.claude/skills/code-wiki/SKILL.md`)

4. **Code blocks:** All steps that modify code have actual code shown
