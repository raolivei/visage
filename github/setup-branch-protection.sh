#!/bin/bash
#
# Setup Branch Protection Rules
#
# This script configures branch protection on 'main' branch to ensure:
# 1. Security scans must pass before merge
# 2. CI checks must pass before merge
# 3. Pull requests are required
# 4. Conversations must be resolved
#
# Prerequisites:
# - GitHub CLI (gh) installed
# - Authenticated with repo admin access
#
# Usage:
#   ./github/setup-branch-protection.sh [--dry-run] [--branch BRANCH]
#
# Options:
#   --dry-run    Show what would be configured without making changes
#   --branch     Branch to protect (default: main)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
error() {
    echo -e "${RED}❌ Error: $1${NC}" >&2
    exit 1
}

info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

success() {
    echo -e "${GREEN}✅ $1${NC}"
}

warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

# Parse arguments
DRY_RUN=false
BRANCH="main"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --branch)
            BRANCH="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--dry-run] [--branch BRANCH]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be configured without making changes"
            echo "  --branch     Branch to protect (default: main)"
            echo "  -h, --help   Show this help message"
            exit 0
            ;;
        *)
            error "Unknown option: $1. Use --help for usage information."
            ;;
    esac
done

# Configuration
REPO_OWNER="raolivei"
REPO_NAME="visage"

if [ -z "$REPO_OWNER" ] || [ -z "$REPO_NAME" ]; then
    error "REPO_OWNER and REPO_NAME must be set"
fi

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    error "GitHub CLI (gh) is not installed. Install it from: https://cli.github.com/"
fi

# Check if authenticated
if ! gh auth status &> /dev/null; then
    error "Not authenticated with GitHub CLI. Run: gh auth login"
fi

info "GitHub CLI authenticated"

# Verify repository exists and user has access
info "Verifying repository access..."
if ! gh repo view "${REPO_OWNER}/${REPO_NAME}" &> /dev/null; then
    error "Repository ${REPO_OWNER}/${REPO_NAME} not found or you don't have access"
fi

# Check if branch exists
info "Checking if branch '${BRANCH}' exists..."
if ! gh api "repos/${REPO_OWNER}/${REPO_NAME}/branches/${BRANCH}" &> /dev/null; then
    error "Branch '${BRANCH}' does not exist in ${REPO_OWNER}/${REPO_NAME}"
fi

success "Repository and branch verified"

# Check if branch protection config file exists
CONFIG_FILE="github/branch-protection-config.json"
if [ ! -f "$CONFIG_FILE" ]; then
    error "Branch protection config file not found: ${CONFIG_FILE}"
fi

# Validate JSON syntax
if ! command -v jq &> /dev/null; then
    warning "jq not installed, skipping JSON validation"
else
    if ! jq empty "$CONFIG_FILE" 2>/dev/null; then
        error "Invalid JSON in ${CONFIG_FILE}"
    fi
    success "JSON syntax validated"
fi

# Detect available status checks from recent workflow runs
info "Detecting available status checks..."
AVAILABLE_CHECKS=$(gh api "repos/${REPO_OWNER}/${REPO_NAME}/commits/${BRANCH}/status" --jq '.statuses[].context' 2>/dev/null | sort -u || echo "")

if [ -n "$AVAILABLE_CHECKS" ]; then
    info "Found status checks:"
    echo "$AVAILABLE_CHECKS" | while read -r check; do
        echo "  - $check"
    done
else
    warning "No status checks found. Workflows may need to run first."
    warning "You can still set up branch protection, but status checks won't be enforced until workflows run."
fi

echo ""
info "Configuration:"
echo "  Repository: ${REPO_OWNER}/${REPO_NAME}"
echo "  Branch: ${BRANCH}"
echo "  Config file: ${CONFIG_FILE}"
if [ "$DRY_RUN" = true ]; then
    echo "  Mode: DRY RUN (no changes will be made)"
fi
echo ""

# Show what will be configured
info "Branch protection rules to apply:"
if command -v jq &> /dev/null; then
    jq -r '.protection | to_entries[] | "  \(.key): \(.value)"' "$CONFIG_FILE" 2>/dev/null || cat "$CONFIG_FILE"
else
    cat "$CONFIG_FILE"
fi
echo ""

# Dry run mode
if [ "$DRY_RUN" = true ]; then
    warning "DRY RUN MODE - No changes will be made"
    echo ""
    info "To apply these rules, run without --dry-run:"
    echo "  $0 --branch ${BRANCH}"
    exit 0
fi

# Confirm before proceeding
echo -n "Apply branch protection rules? (y/N): "
read -r CONFIRM
if [[ ! "$CONFIRM" =~ ^[Yy]$ ]]; then
    info "Cancelled"
    exit 0
fi

# Apply branch protection using GitHub API
info "Applying branch protection rules..."

if gh api \
    --method PUT \
    -H "Accept: application/vnd.github+json" \
    "/repos/${REPO_OWNER}/${REPO_NAME}/branches/${BRANCH}/protection" \
    --input "$CONFIG_FILE" 2>&1; then
    success "Branch protection rules configured successfully!"
else
    error "Failed to configure branch protection. Check your permissions and try again."
fi

echo ""
success "Rules applied:"
echo "  ✅ Pull requests required"
echo "  ✅ Status checks required (Security Scanning, CI)"
echo "  ✅ Branches must be up to date"
echo "  ✅ Conversation resolution required"
echo "  ✅ Linear history enforced"
echo "  ✅ Force push disabled"
echo "  ✅ Branch deletion disabled"
echo "  ✅ Rules apply to administrators"
echo ""
info "PRs cannot be merged until all required status checks pass!"
echo ""
info "View settings at:"
echo "  https://github.com/${REPO_OWNER}/${REPO_NAME}/settings/branches"
echo ""
success "Done!"
