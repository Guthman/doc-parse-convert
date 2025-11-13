# Quick Gitleaks Installation (No Admin Required)

You encountered the admin permissions error with Chocolatey. Here are alternative methods:

## ⚡ Option 1: Scoop (Recommended - No Admin Needed)

Scoop installs to your user directory, no admin rights required!

### Install Scoop (if not installed)
```powershell
# Run in PowerShell (regular, not admin)
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression
```

### Install Gitleaks with Scoop
```powershell
scoop install gitleaks
```

### Verify Installation
```powershell
gitleaks version
```

---

## 📦 Option 2: Manual Installation (Portable)

Download and run Gitleaks without installation:

### Step 1: Download
1. Go to https://github.com/gitleaks/gitleaks/releases/latest
2. Download `gitleaks_*_windows_x64.zip`

### Step 2: Extract
```powershell
# Create a tools directory in your user folder
New-Item -Path "$env:USERPROFILE\tools\gitleaks" -ItemType Directory -Force

# Extract gitleaks.exe to that folder
# (You can do this manually or use PowerShell)
```

### Step 3: Add to PATH (Current Session)
```powershell
$env:PATH += ";$env:USERPROFILE\tools\gitleaks"
```

### Step 4: Add to PATH (Permanent - No Admin)
```powershell
# Add to user PATH (not system PATH, so no admin needed)
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
[Environment]::SetEnvironmentVariable("Path", "$userPath;$env:USERPROFILE\tools\gitleaks", "User")
```

### Step 5: Verify (in new PowerShell window)
```powershell
gitleaks version
```

---

## 🚀 Option 3: Run from Project Directory (Quickest)

Download Gitleaks and place it directly in your project:

### Step 1: Download
1. Go to https://github.com/gitleaks/gitleaks/releases/latest
2. Download `gitleaks_*_windows_x64.zip`
3. Extract `gitleaks.exe` to `D:\PyCharmProjects\guthman-information-extraction-utilities\tools\`

### Step 2: Create Local Tools Directory
```powershell
cd D:\PyCharmProjects\guthman-information-extraction-utilities
New-Item -Path "tools" -ItemType Directory -Force
# Place gitleaks.exe in the tools folder
```

### Step 3: Update Scripts to Use Local Gitleaks

The scan script will automatically find it if you run from the project directory!

---

## ✅ After Installation

### Run Your First Scan
```powershell
cd D:\PyCharmProjects\guthman-information-extraction-utilities
.\scripts\scan-secrets.ps1
```

### Install Pre-commit (Optional)
```powershell
pip install pre-commit
pre-commit install
```

---

## 🎯 Recommended: Use Scoop

**Why Scoop is best for your situation:**
- ✅ No admin rights required
- ✅ Installs to your user directory
- ✅ Easy updates: `scoop update gitleaks`
- ✅ Clean uninstall: `scoop uninstall gitleaks`
- ✅ Manages PATH automatically

**Quick Scoop Install:**
```powershell
# Install Scoop
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
Invoke-RestMethod -Uri https://get.scoop.sh | Invoke-Expression

# Install Gitleaks
scoop install gitleaks

# Verify
gitleaks version

# Run scan
cd D:\PyCharmProjects\guthman-information-extraction-utilities
.\scripts\scan-secrets.ps1
```

---

## ❓ Still Having Issues?

If none of these work, you can:
1. **Ask IT for admin rights** temporarily to install via Chocolatey
2. **Use GitHub Actions only** - The workflow will still scan on push/PR
3. **Manual review** - Carefully review code before committing

The most important protection is the **GitHub Actions workflow** which will catch secrets before they're merged, even if you can't run locally.
