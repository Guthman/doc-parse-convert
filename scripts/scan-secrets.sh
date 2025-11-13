#!/bin/bash
# Shell script to run Gitleaks secret scanning
# Usage: ./scripts/scan-secrets.sh [--verbose] [--all-files]

set -e

VERBOSE=""
ALL_FILES=""
SINCE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --verbose|-v)
            VERBOSE="--verbose"
            shift
            ;;
        --all-files|-a)
            ALL_FILES="--no-git"
            shift
            ;;
        --since)
            SINCE="--log-opts $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "🔍 Scanning for secrets with Gitleaks..."

# Check if Gitleaks is installed
if ! command -v gitleaks &> /dev/null; then
    echo "❌ Gitleaks is not installed!"
    echo ""
    echo "Install options:"
    echo "  macOS: brew install gitleaks"
    echo "  Linux: See https://github.com/gitleaks/gitleaks#installing"
    echo ""
    exit 1
fi

# Run Gitleaks
gitleaks detect \
    --source . \
    --config .gitleaks.toml \
    --report-path gitleaks-report.json \
    $VERBOSE \
    $ALL_FILES \
    $SINCE

if [ $? -eq 0 ]; then
    echo "✅ No secrets detected!"
elif [ $? -eq 1 ]; then
    echo "❌ Secrets detected! Check gitleaks-report.json for details."

    # Display summary if jq is available
    if command -v jq &> /dev/null && [ -f gitleaks-report.json ]; then
        echo ""
        COUNT=$(jq length gitleaks-report.json)
        echo "Found $COUNT potential secret(s):"
        jq -r '.[] | "  - \(.File):\(.StartLine) - \(.RuleID)"' gitleaks-report.json
    fi

    exit 1
else
    echo "❌ Gitleaks exited with code $?"
    exit $?
fi
