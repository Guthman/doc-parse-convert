# Release Process

This document describes how to release a new version of `doc_parse_convert` to PyPI.

## Automated Publishing Setup

The project uses GitHub Actions to automatically publish to PyPI when a new release is created.

### One-Time Setup

1. **Create PyPI API Token**
   - Go to https://pypi.org/manage/account/token/
   - Click "Add API token"
   - Name: `doc_parse_convert-github-actions`
   - Scope: "Entire account" (or specific to this project once first published)
   - Copy the token (starts with `pypi-`)

2. **Add Token to GitHub Secrets**
   - Go to your GitHub repository settings
   - Navigate to: Settings → Secrets and variables → Actions
   - Click "New repository secret"
   - Name: `PYPI_API_TOKEN`
   - Value: Paste the PyPI token
   - Click "Add secret"

3. **[Optional] Setup TestPyPI for Testing**
   - Create account at https://test.pypi.org/account/register/
   - Create API token at https://test.pypi.org/manage/account/token/
   - Add to GitHub Secrets as `TEST_PYPI_API_TOKEN`

## Release Workflow

### 1. Prepare the Release

```bash
# Ensure you're on the master branch and up to date
git checkout master
git pull origin master

# Run tests to ensure everything works
pytest

# Update version in setup.py
# Edit: version="0.5.5"  # Example: increment version
```

### 2. Update Changelog (Recommended)

Create/update `CHANGELOG.md` with:
```markdown
## [0.5.5] - 2025-XX-XX
### Added
- New feature X

### Fixed
- Bug fix Y

### Changed
- Improvement Z
```

### 3. Commit Version Changes

```bash
git add setup.py CHANGELOG.md
git commit -m "chore: bump version to 0.5.5"
git push origin master
```

### 4. Create Git Tag

```bash
# Create annotated tag
git tag -a v0.5.5 -m "Release version 0.5.5"

# Push tag to GitHub
git push origin v0.5.5
```

### 5. Create GitHub Release

**Option A: Via GitHub Web Interface**
1. Go to your repository on GitHub
2. Click "Releases" → "Draft a new release"
3. Choose tag: `v0.5.5`
4. Release title: `v0.5.5` or `Release 0.5.5`
5. Description: Copy relevant section from CHANGELOG.md
6. Click "Publish release"

**Option B: Via GitHub CLI**
```bash
gh release create v0.5.5 \
  --title "Release 0.5.5" \
  --notes "Release notes here"
```

### 6. Automatic Publishing

Once the release is published:
- GitHub Actions automatically triggers
- Builds source and wheel distributions
- Uploads to PyPI
- Check progress: Actions tab on GitHub

### 7. Verify Publication

```bash
# Wait a few minutes, then verify
pip install --upgrade doc_parse_convert

# Check version
python -c "import doc_parse_convert; print(doc_parse_convert.__version__)"
```

## Testing Releases (TestPyPI)

Before publishing to PyPI, test on TestPyPI:

```bash
# Manually trigger TestPyPI workflow
# Go to: Actions → Publish to TestPyPI → Run workflow

# Or push to develop branch (if configured)
git checkout -b develop
git push origin develop
```

Install from TestPyPI:
```bash
pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  doc_parse_convert
```

Note: `--extra-index-url` is needed because dependencies come from PyPI.

## Manual Publishing (Fallback)

If GitHub Actions fails:

```bash
# Install tools
pip install build twine

# Build
python -m build

# Upload
twine upload dist/*
```

## Versioning Guidelines

Follow Semantic Versioning (semver):
- **MAJOR** (X.0.0): Breaking changes
- **MINOR** (0.X.0): New features, backward compatible
- **PATCH** (0.0.X): Bug fixes, backward compatible

Examples:
- `0.5.4` → `0.5.5`: Bug fix
- `0.5.5` → `0.6.0`: New feature
- `0.6.0` → `1.0.0`: Breaking change

## Troubleshooting

**GitHub Action fails with "403 Forbidden"**
- Check that `PYPI_API_TOKEN` secret is set correctly
- Verify token hasn't expired
- Ensure token has correct permissions

**Package already exists**
- You cannot re-upload the same version
- Increment version number and try again

**Tests fail during build**
- GitHub Actions doesn't run tests by default (add if needed)
- Always run `pytest` locally before releasing

**Import error after installation**
- Check package structure with `python -m build`
- Verify `find_packages()` includes all modules
- Test in clean virtual environment first

## Pre-Release Checklist

- [ ] All tests pass locally (`pytest`)
- [ ] Version bumped in `setup.py`
- [ ] CHANGELOG.md updated (if exists)
- [ ] Changes committed and pushed
- [ ] Git tag created and pushed
- [ ] GitHub release created
- [ ] GitHub Action completed successfully
- [ ] Package installable: `pip install doc_parse_convert`
- [ ] Version correct: Check on https://pypi.org/project/doc_parse_convert/

## Rollback

If you need to remove a bad release:
1. PyPI doesn't allow deletion of releases (prevents dependency breakage)
2. Instead, publish a new patch version with fixes
3. Mark the bad version as "yanked" on PyPI (prevents new installs but allows existing)
