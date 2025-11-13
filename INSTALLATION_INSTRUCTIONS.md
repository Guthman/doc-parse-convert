# Installation Instructions for Gitleaks Secret Scanning

## ⚠️ Manual Installation Required

The Gitleaks installation requires administrator privileges. Please follow these steps:

### Option 1: Install with Chocolatey (Recommended)

Open **PowerShell as Administrator** and run:

```powershell
choco install gitleaks -y
```

Verify installation:
```powershell
gitleaks version
```

### Option 2: Install with Scoop

Open **PowerShell** (no admin required) and run:

```powershell
scoop install gitleaks
```

### Option 3: Manual Download

1. Visit https://github.com/gitleaks/gitleaks/releases
2. Download `gitleaks_*_windows_x64.zip` for your platform
3. Extract `gitleaks.exe` to a folder (e.g., `C:\tools\gitleaks\`)
4. Add that folder to your system PATH
5. Verify: `gitleaks version`

## After Installing Gitleaks

### 1. Install Pre-commit (Optional)

```bash
pip install pre-commit
pre-commit install
```

This enables automatic secret scanning before each commit.

### 2. Run Your First Scan

**Windows:**
```powershell
.\scripts\scan-secrets.ps1
```

**Linux/Mac:**
```bash
chmod +x scripts/scan-secrets.sh
./scripts/scan-secrets.sh
```

### 3. Set Up Environment Variables

1. Copy `.env.example` to `.env`:
   ```powershell
   Copy-Item .env.example .env
   ```

2. Edit `.env` and add your actual credentials (this file is gitignored)

## What's Already Set Up

✅ **Gitleaks configuration** (`.gitleaks.toml`) - Configured with:
  - Default secret detection rules
  - Custom rules for Google Cloud and Jina API
  - Allowlists for documentation and examples

✅ **Pre-commit configuration** (`.pre-commit-config.yaml`) - Ready to use once pre-commit is installed

✅ **GitHub Actions** (`.github/workflows/gitleaks.yml`) - Automatic scanning on push/PR

✅ **Helper scripts** - Easy-to-use scanning scripts for Windows and Linux/Mac

✅ **Updated .gitignore** - Prevents committing:
  - `.env` files
  - Service account JSON files
  - Private keys
  - Gitleaks reports

✅ **Documentation**:
  - `SECURITY.md` - Security policy and best practices
  - `GITLEAKS_SETUP.md` - Detailed setup guide
  - `.env.example` - Template for environment variables

## Quick Reference

### Scan Commands

```powershell
# Quick scan
.\scripts\scan-secrets.ps1

# Verbose output
.\scripts\scan-secrets.ps1 -Verbose

# Scan all files (including uncommitted)
.\scripts\scan-secrets.ps1 -AllFiles
```

### What to Do If Secrets Are Found

1. ❌ **DO NOT commit** the code
2. 🔍 Review `gitleaks-report.json` to see what was detected
3. 🔧 Move secrets to environment variables or `.env` file
4. ♻️ If already committed, rotate the credential and clean git history
5. ✅ Re-run the scan to verify

## Testing the Integration

After installing Gitleaks, test it works:

```powershell
# Run a scan
.\scripts\scan-secrets.ps1

# Should output: "✅ No secrets detected!"
```

## Next Steps

1. ✅ Install Gitleaks (see options above)
2. ✅ Run your first scan
3. ✅ Install pre-commit for automatic scanning
4. ✅ Set up your `.env` file
5. ✅ Review `SECURITY.md` for best practices

## Need Help?

- **Gitleaks Setup**: See `GITLEAKS_SETUP.md`
- **Security Policy**: See `SECURITY.md`
- **Gitleaks Documentation**: https://github.com/gitleaks/gitleaks
