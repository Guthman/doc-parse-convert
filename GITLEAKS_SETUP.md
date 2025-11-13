# Gitleaks Setup Guide

This guide will help you set up Gitleaks to prevent committing secrets to the repository.

## Quick Start

### 1. Install Gitleaks

**Windows (Chocolatey - Recommended):**
```powershell
choco install gitleaks
```

**Windows (Scoop):**
```powershell
scoop install gitleaks
```

**Windows (Manual):**
1. Download the latest release from https://github.com/gitleaks/gitleaks/releases
2. Extract `gitleaks.exe` to a folder in your PATH (e.g., `C:\Program Files\gitleaks\`)
3. Verify installation: `gitleaks version`

**macOS:**
```bash
brew install gitleaks
```

**Linux:**
```bash
# Download and install
wget https://github.com/gitleaks/gitleaks/releases/download/v8.21.2/gitleaks_8.21.2_linux_x64.tar.gz
tar -xzf gitleaks_8.21.2_linux_x64.tar.gz
sudo mv gitleaks /usr/local/bin/
```

### 2. Install Pre-commit (Optional but Recommended)

This will automatically scan for secrets before each commit.

```bash
pip install pre-commit
pre-commit install
```

### 3. Run Your First Scan

**Windows:**
```powershell
.\scripts\scan-secrets.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/scan-secrets.sh
./scripts/scan-secrets.sh
```

## Usage

### Automatic Scanning (Pre-commit Hook)

Once pre-commit is installed, Gitleaks will run automatically before each commit:

```bash
git add .
git commit -m "Your commit message"
# Gitleaks runs automatically here
```

If secrets are detected, the commit will be blocked and you'll see a report.

### Manual Scanning

**Scan entire repository:**
```powershell
# Windows
.\scripts\scan-secrets.ps1

# Linux/Mac
./scripts/scan-secrets.sh
```

**Scan with verbose output:**
```powershell
# Windows
.\scripts\scan-secrets.ps1 -Verbose

# Linux/Mac
./scripts/scan-secrets.sh --verbose
```

**Scan all files (including uncommitted):**
```powershell
# Windows
.\scripts\scan-secrets.ps1 -AllFiles

# Linux/Mac
./scripts/scan-secrets.sh --all-files
```

### Direct Gitleaks Commands

```bash
# Scan the entire repository
gitleaks detect --source . --config .gitleaks.toml

# Scan with verbose output
gitleaks detect --source . --config .gitleaks.toml --verbose

# Scan specific commits
gitleaks detect --source . --log-opts "origin/main..HEAD"

# Scan without git (all files)
gitleaks detect --source . --no-git
```

## What If Secrets Are Found?

1. **Review the findings** in `gitleaks-report.json`
2. **Remove the secret** from your code
3. **Use environment variables** instead:
   ```python
   import os
   api_key = os.environ.get('JINA_API_KEY')
   ```
4. **Copy `.env.example` to `.env`** and add your secrets there (`.env` is gitignored)
5. **If the secret was already committed**, you need to:
   - Remove it from git history (use `git filter-branch` or BFG Repo-Cleaner)
   - Rotate/regenerate the credential
   - Never push the commit with the secret

## Handling False Positives

If Gitleaks flags something that isn't a secret:

1. Verify it's truly not sensitive
2. Edit `.gitleaks.toml` to add an allowlist entry:

```toml
[allowlist]
regexes = [
  '''your-false-positive-pattern'''
]

# Or allowlist specific files
paths = [
  '''tests/test_example\.py'''
]
```

## CI/CD Integration

The repository includes GitHub Actions workflow (`.github/workflows/gitleaks.yml`) that automatically scans:
- All pushes to main/master/develop branches
- All pull requests
- Manual workflow runs

## Configuration

Gitleaks configuration is in `.gitleaks.toml`. It includes:

- **Default rules**: All standard Gitleaks secret detection patterns
- **Custom rules**:
  - Google Service Account JSON detection
  - Jina API key detection
  - GCP Project ID detection (for review)
- **Allowlists**: Documentation files, examples, and known false positives

## Best Practices

1. ✅ Use `.env` files for local secrets (already in `.gitignore`)
2. ✅ Use environment variables in code
3. ✅ Run scans before pushing to remote
4. ✅ Keep service account files outside the repository
5. ✅ Use placeholders in documentation (e.g., `your-api-key`)
6. ❌ Never hardcode credentials
7. ❌ Never commit `.env` files
8. ❌ Never commit service account JSON files

## Troubleshooting

**"gitleaks: command not found"**
- Gitleaks is not installed or not in PATH
- Follow installation instructions above

**Pre-commit hook not running**
- Run `pre-commit install` to set up the hook
- Verify with `pre-commit run --all-files`

**Too many false positives**
- Review and update `.gitleaks.toml` allowlist
- Use more specific patterns
- Consider using `--verbose` to see detection details

## Additional Resources

- [Gitleaks Documentation](https://github.com/gitleaks/gitleaks)
- [Pre-commit Documentation](https://pre-commit.com/)
- [SECURITY.md](./SECURITY.md) - Full security policy
