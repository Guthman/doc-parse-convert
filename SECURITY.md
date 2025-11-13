i# Security Policy

## Secret Scanning

This repository uses [Gitleaks](https://github.com/gitleaks/gitleaks) to prevent accidentally committing secrets and credentials.

### Automated Scanning

1. **Pre-commit Hook**: Secrets are scanned automatically before each commit (if pre-commit is installed)
2. **GitHub Actions**: All pushes and pull requests are automatically scanned
3. **Manual Scanning**: Run `.\scripts\scan-secrets.ps1` (Windows) or `./scripts/scan-secrets.sh` (Linux/Mac)

### Setup Instructions

#### Install Gitleaks

**Windows (using Chocolatey):**
```powershell
choco install gitleaks
```

**Windows (using Scoop):**
```powershell
scoop install gitleaks
```

**macOS:**
```bash
brew install gitleaks
```

**Linux:**
```bash
# Download latest release
wget https://github.com/gitleaks/gitleaks/releases/download/v8.21.2/gitleaks_8.21.2_linux_x64.tar.gz
tar -xzf gitleaks_8.21.2_linux_x64.tar.gz
sudo mv gitleaks /usr/local/bin/
```

#### Install Pre-commit (Optional but Recommended)

```bash
pip install pre-commit
pre-commit install
```

This will automatically run Gitleaks before every commit.

### Running Manual Scans

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

**Scan only recent commits:**
```bash
# Linux/Mac
./scripts/scan-secrets.sh --since "origin/main..HEAD"
```

### Handling Secrets

If Gitleaks detects a secret:

1. **Never commit the secret** - Review the finding in `gitleaks-report.json`
2. **Remove the secret** from the file
3. **Use environment variables** or `.env` file instead (ensure `.env` is in `.gitignore`)
4. **Rotate the credential** if it was accidentally committed
5. **Add to allowlist** if it's a false positive (edit `.gitleaks.toml`)

### Best Practices

1. **Use environment variables** for all sensitive data:
   ```python
   import os
   api_key = os.environ.get('JINA_API_KEY')
   ```

2. **Use `.env` files** for local development (already in `.gitignore`):
   ```
   JINA_API_KEY=your_key_here
   GCP_SERVICE_ACCOUNT=path/to/service-account.json
   ```

3. **Use configuration objects** that load from environment:
   ```python
   config = ProcessingConfig(
       project_id=os.environ.get('GCP_PROJECT_ID'),
       jina_api_token=os.environ.get('JINA_API_KEY')
   )
   ```

4. **Never hardcode** credentials in:
   - Source code
   - Configuration files (unless using placeholders)
   - Test files (use mocks or environment variables)
   - Documentation (use placeholder examples like `your-api-key`)

### Configuration

The Gitleaks configuration is in `.gitleaks.toml`. It includes:
- Default Gitleaks rules
- Custom rules for Google Service Accounts and Jina API keys
- Allowlist for documentation and example files
- False positive patterns

### False Positives

If you encounter false positives:

1. Verify it's truly not a secret
2. Add to the allowlist in `.gitleaks.toml`:
   ```toml
   [allowlist]
   regexes = [
     '''your-pattern-here'''
   ]
   ```

### Reporting Security Issues

If you discover a security vulnerability, please email the maintainer instead of using the issue tracker.

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.5.x   | :white_check_mark: |
| < 0.5   | :x:                |
