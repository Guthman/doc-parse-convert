# PowerShell script to run Gitleaks secret scanning
# Usage: .\scripts\scan-secrets.ps1

param(
    [switch]$Verbose,
    [switch]$AllFiles,
    [string]$Since = ""
)

$ErrorActionPreference = "Stop"

Write-Host "Scanning for secrets with Gitleaks..."

# Check if Gitleaks is installed
if (-not (Get-Command gitleaks -ErrorAction SilentlyContinue)) {
    Write-Host "Gitleaks is not installed!"
    Write-Host ""
    Write-Host "Install options:"
    Write-Host "  1. Using Chocolatey (recommended):"
    Write-Host "     choco install gitleaks"
    Write-Host ""
    Write-Host "  2. Using Scoop:"
    Write-Host "     scoop install gitleaks"
    Write-Host ""
    Write-Host "  3. Download from GitHub:"
    Write-Host "     https://github.com/gitleaks/gitleaks/releases"
    Write-Host ""
    exit 1
}

# Build command arguments
$args = @("detect", "--source", ".", "--config", ".gitleaks.toml", "--report-path", "gitleaks-report.json")

if ($Verbose) {
    $args += "--verbose"
}

if ($AllFiles) {
    $args += "--no-git"
}

if ($Since) {
    $args += "--log-opts"
    $args += $Since
}

# Run Gitleaks
Write-Host "Running: gitleaks $($args -join ' ')"
& gitleaks $args

if ($LASTEXITCODE -eq 0) {
    Write-Host "No secrets detected!"
} elseif ($LASTEXITCODE -eq 1) {
    Write-Host "Secrets detected! Check gitleaks-report.json for details."

    # Display summary if report exists
    if (Test-Path "gitleaks-report.json") {
        $report = Get-Content "gitleaks-report.json" | ConvertFrom-Json
        Write-Host ""
        Write-Host "Found $($report.Count) potential secret(s):"
        foreach ($finding in $report) {
            Write-Host "  - $($finding.File):$($finding.StartLine) - $($finding.RuleID)"
        }
    }

    exit 1
} else {
    Write-Host "Gitleaks exited with code $LASTEXITCODE"
    exit $LASTEXITCODE
}
