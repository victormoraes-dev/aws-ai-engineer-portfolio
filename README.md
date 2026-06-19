# ai-engineer-portfolio

Welcome to the AI Engineer portfolio monorepo for Victor Moraes.

This repository showcases projects, experiments, and reusable tooling built for AI engineering, full-stack applications, and machine learning workflows.

## Overview

- **apps/**: Deployable applications (API, web, mobile)
- **packages/**: Shared libraries and internal tooling
- **libs/**: AI/ML-specific modules and agent utilities
- **docs/**: Architecture Decision Records, API docs, and guides
- **tools/**: CLI scripts, automation, and DevOps assets
- **tests/**: Integration and end-to-end tests

## Quick Start

```bash
git clone https://github.com/victormoraes-dev/aws-ai-engineer-portfolio.git
cd aws-ai-engineer-portfolio

# Install dependencies and run setup
./tools/scripts/setup.sh
```

## Development

- Follow the conventions in [docs/guides/CONTRIBUTING.md](docs/guides/CONTRIBUTING.md).
- Use the workspace configuration in [packages/config](packages/config) for shared linting, formatting, and build settings.
- Open Pull Requests against the `main` branch and ensure CI checks pass.

## License

[MIT](LICENSE) © Your Name
