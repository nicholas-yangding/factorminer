# Code Wiki Skill Design

## Overview

A skill for building and maintaining a code knowledge base (Wiki) from source code repositories. Combines automatic extraction of code structure with AI-driven interpretation of design intent.

## Trigger Conditions

Use when user mentions:
- "创建代码知识库" / "build code wiki"
- "理解这个仓库" / "understand this codebase"
- "查一下代码" / "look up code"
- "检查知识库健康" / "lint wiki"
- "更新知识库" / "update wiki"

## Three-Layer Architecture

| Layer | Location | Content | Management |
|-------|----------|---------|------------|
| **Sources** | `code-wiki/sources/` | Code file snapshots (versioned) | Automated |
| **Wiki** | `code-wiki/` | Module pages, concept pages, design decision pages | AI maintained |
| **Index** | `code-wiki/index.md` | Content index + search entry | AI updated |

## Directory Structure

```
code-wiki/
├── index.md              # Content index, organized by module/concept
├── log.md                # Operation log (build, query, maintenance)
├── modules/              # Module pages
│   ├── agent.md          # agent module: responsibilities, key classes
│   ├── core-parser.md    # core/parser module
│   └── ...
├── concepts/             # Concept pages
│   ├── expression-tree.md # Expression tree design
│   ├── ralph-loop.md     # Ralph loop mechanism
│   └── ...
├── design/               # Design decision pages
│   ├── typed-dsl.md      # DSL design decisions
│   └── ...
└── sources/              # Code snapshots (optional)
```

## Core Operations

### Build

1. Auto-extract file tree, function signatures, class relationships
2. AI analyzes design intent of key modules
3. Generate/update `modules/`, `concepts/`, `design/` pages
4. Update `index.md` and `log.md`

### Query

1. Read Wiki to locate relevant modules
2. Read source code with Wiki design context
3. Synthesize answer; insights not in Wiki can be written back

### Lint

1. Detect outdated content (source deleted but Wiki still exists)
2. Detect orphaned pages (no cross-references)
3. Detect knowledge gaps (important modules without pages)

## Page Types

| Type | Naming | Content |
|------|--------|---------|
| Module page | `modules/{module-name}.md` | Module purpose, key classes, relationships |
| Concept page | `concepts/{concept-name}.md` | Design patterns, algorithms, architecture |
| Design page | `design/{decision-name}.md` | Design decisions, trade-offs, rationale |

## Query Flow

Always combine Wiki with source code reading:
1. Search Wiki first for relevant context
2. Read source code with that context
3. Answer synthetically, with Wiki references

## Hybrid Extraction

Initial build uses mixed approach:
- **Auto-extraction**: File tree, function signatures, class relationships (globs/grep)
- **AI interpretation**: Focus on design intent of key modules

## Implementation Location

Skill file: `~/.claude/skills/code-wiki/SKILL.md`
