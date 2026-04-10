# parkour-mcp development tasks

# Pack Claude Desktop Extension bundle
pack:
    mkdir -p dist
    npx @anthropic-ai/mcpb pack . dist/parkour-mcp.mcpb

# Run unit tests (mocked, excludes live)
test *args:
    uv run pytest {{args}}

# Run live integration tests
test-live:
    uv run pytest -m live

# Run performance regression tests against captured fixtures
test-perf:
    uv run pytest -m perf

# Run the pipeline benchmark (pass --update-baselines or --capture-fixtures as args)
benchmark *args:
    uv run python3 scripts/benchmark_pipeline.py {{args}}

# Regenerate README examples
readme:
    uv run python3 scripts/regenerate_readme_examples.py
