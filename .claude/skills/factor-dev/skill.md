---
name: factor-dev
description: Factor development for the featureengineering project. Provides architecture, data catalog, development patterns, constraints, and pending factor ideas. Always load before writing new factors.
---

# Factor Development Skill

Use this skill when developing new Alpha factors for the LingQi featureengineering pipeline.
Load all reference documents in `references/` before writing any factor code.

## Reference Documents

| Document | Content |
|----------|---------|
| [constraints.md](references/constraints.md) | **Hard constraints** — stock pool, frequency, NaN rate, phases, output alignment |
| [project-architecture.md](references/project-architecture.md) | Codebase architecture, key modules, data flow |
| [data-catalog.md](references/data-catalog.md) | Complete upstream data inventory: files, columns, time ranges |
| [factor-patterns.md](references/factor-patterns.md) | Factor registration, compute patterns, conventions, checklist |
| [pending-ideas.md](references/pending-ideas.md) | Factors NOT yet implemented, prioritized by data readiness |
| [utility-api.md](references/utility-api.md) | Utility functions and DataRepository API reference |

## Quick Start

1. **Read `constraints.md` first** — these are hard rules, violations mean the factor is rejected
2. Read `project-architecture.md` to understand the codebase
3. Read `data-catalog.md` to find relevant data sources
4. Read `factor-patterns.md` for the exact code pattern to follow
5. Check `pending-ideas.md` for pre-scoped factors ready to implement
6. Reference `utility-api.md` for available helper functions

## Validation

After building factors, run the validation script:
```bash
python scripts/validate_factors.py
```
This checks: last-date alignment, NaN rate from 2020-01-01, stock count, and frequency.
