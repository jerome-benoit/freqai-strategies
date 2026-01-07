---
agent: build
description: Commit changes with a concise and precise message
---

# Commit Command

Create well-structured commits following Conventional Commits format.

## Workflow

1. **Analyze changes**
   - Run: `git status`, `git diff --cached`, `git diff`, `git log -5 --oneline`
   - Stage needed files with `git add`
   - Exclude secrets and generated/large artifacts

2. **Craft message** following format below

3. **Confirm and commit**
   - Show proposed message and ask confirmation
   - Run: `git commit -m "<message>"` then `git status`
   - Do not commit without confirmation
   - Do not force push or amend unless requested
   - If a hook fails, report and suggest fixes

## Message Format

**Structure**: `<type>(<scope>): <summary>`

**Type** (required):

- `feat`, `fix`, `refactor`, `perf`, `docs`, `style`, `test`, `chore`, `ci`,
  `build`, `revert`

**Scope** (optional):

- Strategy name: `quickadapter`, `ReforceXY`
- Component name if cross-strategy

**Summary** (required):

- ≤ 72 chars, imperative mood
- Focus on WHAT changed (the diff shows HOW)

**Body** (optional):

- Explain WHY (impact, context, reasoning)
- Add with second `-m` flag: `git commit -m "<summary>" -m "<body>"`
- Reference issues if applicable

**Breaking changes**:

- Add `!` after scope: `feat(quickadapter)!: change API`

## Examples

```
feat(quickadapter): add EarlyStopping callback for XGBoost 3.x API
fix(ReforceXY): correct LightGBM validation set naming in pruning
refactor: remove redundant comments in regressor initialization
```
