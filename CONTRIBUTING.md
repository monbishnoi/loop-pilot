# Contributing to Loop Pilot

Thanks for your interest in contributing! Loop Pilot is a young project and contributions are welcome.

## Getting Started

```bash
git clone https://github.com/monbishnoi/loop-pilot.git
cd loop-pilot
npm install
npm run build
npm test
```

## Development

Loop Pilot is written in TypeScript and uses Node.js's built-in test runner.

```bash
npm run build    # Compile TypeScript
npm test         # Build + run tests
npm run check    # Type-check without emitting
```

## What We're Looking For

### Harness Adapters
Loop Pilot currently ships with a JSONL event log parser (built for Cal Gateway). We'd love adapters for:
- LangChain / LangGraph trace format
- CrewAI execution logs
- AutoGen conversation logs
- Custom agent harnesses

An adapter is a parser that reads your harness's log format and outputs `ToolCallEpisode[]`. See `src/adapters/jsonl-events/parser.ts` for the reference implementation.

### Embedding Providers
Currently supported: HTTP endpoint, CLI command. Potential additions:
- OpenAI embeddings provider
- Ollama provider
- Direct ONNX runtime provider

### Benchmarks & Data
If you run Loop Pilot on your harness and collect benchmark results, we'd love to see them. Open an issue with your findings.

## Code Style

- TypeScript strict mode
- Explicit types on public interfaces
- No `any` — use `unknown` and narrow
- Descriptive variable names over comments

## Pull Requests

1. Fork the repo
2. Create a feature branch (`git checkout -b feat/my-adapter`)
3. Write tests for new functionality
4. Ensure `npm test` passes
5. Open a PR with a clear description of what and why

## Issues

Use issues for:
- Bug reports (include Node version, OS, and reproduction steps)
- Feature requests (describe the use case, not just the solution)
- Questions about integration with your harness

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
