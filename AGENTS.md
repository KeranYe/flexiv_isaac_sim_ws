# AGENTS.md

This file is the primary shared configuration file for AI programming tools working on this repository.

`README.md` and the `docs/` directory already contain the project introduction, directory structure, build commands, test commands, verification commands, and coding style. This file only records the collaboration rules that AI agents must prioritize when executing tasks.

## Tool Entrypoints

- `CLAUDE.md` and `GEMINI.md` reference this file via `@AGENTS.md`.
- `.cursor/rules/main.mdc` is a symlink pointing to this file.
- `.github/copilot-instructions.md` is a symlink pointing to this file.
- Tool-specific supplements should only be written in their respective configuration files; do not duplicate shared rules here.

## Basic Principles

- Read the relevant code, `README.md`, `docs/`, module documentation, and adjacent tests before making changes.
- Prioritize minimal necessary changes. Do not expand the scope or refactor opportunistically.
- Follow the existing architecture, naming conventions, error handling, logging, and testing styles.
- Do not introduce new dependencies unless the task explicitly requires them or there is a well-justified reason.
- Do not modify public APIs, configuration formats, data formats, or protocols unless the task explicitly requires it.
- Do not modify generated files, build artifacts, or third-party code unless the task explicitly requires it.
- When referencing code locations, use the Markdown link format: `[filename.ext:line](relative/path#Lline)`.
- Use English as the default language. Only switch to Chinese when explicitly instructed. 

## Workflow

For non-trivial modifications:

1. Assess the scope of impact.
2. Formulate a brief plan.
3. Modify the code.
4. Run the minimum relevant verification.
5. Summarize the changes, verification results, and remaining risks.

When verification cannot be run, you must clearly state the attempted commands, the reasons for failure, and the unverified risks.

## Testing and Quality

- Behavioral changes must be accompanied by new or updated tests.
- Bug fixes should include regression tests.
- Do not weaken test assertions just to pass tests.
- Do not use mocks to conceal design issues.
- Do not claim to have run commands that were not actually executed.
- ROS 2 builds and tests must be executed from the repository root. Do not enter individual package directories to generate workspace artifacts.

## Architecture Constraints

- Prioritize reusing existing modules. Do not create parallel implementations.
- Core logic should be separated from IO, networking, databases, hardware, and UI.
- Do not introduce circular dependencies by calling across layers.
- Do not hard-code environment paths, device names, IP addresses, accounts, or keys in core modules.
- Do not modify module behavior, topic/service interfaces, URDF, message definitions, or behavior tree XML without reading the relevant README/doc first.
- Do not arbitrarily modify vendored code under `third_party/` unless the task explicitly requires it.

## Security Boundaries

Committing passwords, tokens, private keys, real account details, sensitive customer information, unmasked logs, or large temporary files is prohibited.

## Known Pitfalls

- This repository is a ROS 2 Humble colcon workspace, but business packages are not located under the default `src/` directory. Builds, tests, and pre-execution workspace sourcing should all be performed from the repository root.
- Complex processes and project explanations are maintained in `README.md`, `docs/`, and module documentation. Do not copy installation tutorials, long-running procedures, or background explanations back into this file.
- Only add rules derived from real issues that are long-lasting and verifiable. Do not pre-stack outdated or speculative rules.

## Retrospective

If rework occurs, evaluate whether this file needs to be supplemented. Only add minimal rules derived from real issues that are long-lasting and verifiable.