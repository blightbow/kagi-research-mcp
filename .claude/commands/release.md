---
description: Cut a new release (stage bump, review, commit, tag). Push stays manual.
---

Walk the user through the parkour-mcp release flow. This stages and
tags locally; the user pushes when ready (yubikey required).

## Step 1: Preflight

Run these in parallel and report briefly:

- `git status` (working tree must be clean)
- `git log origin/main..HEAD --oneline` (show the commits that will ship)
- `uv run cz bump --get-next --yes` (next version from commits since last tag)

Abort if working tree is dirty or no commits ahead of origin/main.
Flag to the user instead of proceeding.

Also spot-check that every `feat:` / `fix:` / `refactor:` / `perf:`
commit in the range carries a `Why:` trailer. Run
`git log origin/main..HEAD --format='%h %s%n%(trailers:key=Why,only)'`
and flag any commits that are missing one. `Why:` trailers are the
source of release-note prose; commits without them fall back to the
bare subject which produces weaker changelog entries.

## Step 2: Preview

Show the user what will land:

```
uv run cz bump --dry-run --yes
git cliff --tag v<NEXT> --unreleased
```

where `<NEXT>` is the version from step 1. Together these preview the
version bump and the assembled CHANGELOG entry without writing
anything. Pause; let the user approve or ask for commit-message edits
(via `git commit --amend` or `git rebase -i`) before you stage.

If the user wants a public RC (finals-only is the default), they can
ask for it explicitly. The RC equivalent of step 3 is:

```
uv run cz bump --version-files-only --yes --prerelease rc
```

commitizen's `pep440` version scheme emits `1.2.0rc1` (not the SemVer
dashed form that would break `uv build`).

## Step 3: Stage the bump

```
uv run cz bump --version-files-only --yes
```

commitizen bumps `project.version` in `pyproject.toml` only. It does
NOT commit or tag. `--yes` is safe here because the user saw the
preview in step 2 and approved.

Then run the downstream steps that populate the rest of the release
commit:

```
NEXT=$(uv run python3 -c "import tomllib; print(tomllib.load(open('pyproject.toml','rb'))['project']['version'])")
git cliff --tag "v$NEXT" --unreleased --prepend CHANGELOG.md
uv run python3 scripts/sync_versions.py
```

- `git cliff --tag --unreleased --prepend` parses Conventional Commits
  from git log, extracts `Why:` trailers as user-facing narrative, and
  prepends an assembled `## [$NEXT] <date>` section to `CHANGELOG.md`.
- `sync_versions.py` mirrors the new `project.version` into
  `manifest.json` (translated to strict SemVer for Claude Desktop) and
  `server.json` (PEP 440 verbatim for MCP Registry).

## Step 4: Review

- Show `git status` to confirm the changed set (pyproject, CHANGELOG,
  manifest, server).
- Show `git diff` for the new CHANGELOG.md entry specifically.
- git-cliff renders the `Why:` trailer as the bullet text. If any
  trailer was written loosely at commit time, this is the last chance
  to tighten it. Options: amend the commit's trailer and re-run
  `git cliff --tag --unreleased --prepend CHANGELOG.md` (after
  reverting the previous CHANGELOG edit), or hand-edit CHANGELOG.md
  directly. The latter is simpler for small wording tweaks.
- Sanity: every `feat:` / `fix:` / `refactor:` / `perf:` commit in
  the range should produce a bullet in the matching section. Flag any
  that are missing (probably means the commit's type didn't match any
  `commit_parsers` rule in `pyproject.toml`).

## Step 5: Commit, then tag

Order matters: commit before tag so the tag points at the release
commit, not its parent.

```
git add -A
git commit -m "release: v<NEXT>"
just tag v<NEXT>
```

`just tag` runs `sync_versions.py --check`, the mocked test suite (with
ruff lint), and the live test suite before creating the annotated tag.
Expect ~1-2 minutes for live tests.

## Step 6: Hand off

Remind the user:

> Tag created locally. Push with:
>
>     git push origin main --follow-tags
>
> The release workflow fires on tag push and handles: uv build,
> PyPI OIDC publish, mcpb pack, GitHub Release creation (body from
> CHANGELOG.md slice), server.json mcpb asset coordinates, MCP Registry
> publish. Watch the run at:
> https://github.com/blightbow/parkour-mcp/actions

Do not push. Do not trigger the workflow. The user pushes.
