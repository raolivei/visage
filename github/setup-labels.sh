#!/bin/bash

# Setup GitHub Labels
# This script creates standardized labels in a GitHub repository
# Labels are aligned with branch naming conventions (feature/, fix/, infra/, etc.)

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get repository name from git remote or prompt
REPO_NAME="${1:-}"
if [ -z "$REPO_NAME" ]; then
    # Try to get from git remote
    if git remote get-url origin &>/dev/null; then
        REPO_URL=$(git remote get-url origin)
        if [[ "$REPO_URL" =~ github.com[:/]([^/]+)/([^/]+)(\.git)?$ ]]; then
            REPO_OWNER="${BASH_REMATCH[1]}"
            REPO_NAME="${BASH_REMATCH[2]%.git}"
            REPO_NAME="${REPO_OWNER}/${REPO_NAME}"
        fi
    fi
    
    if [ -z "$REPO_NAME" ]; then
        echo -e "${RED}Error: Repository name not provided and could not be determined from git remote${NC}"
        echo "Usage: $0 [OWNER/REPO_NAME]"
        echo "Example: $0 raolivei/visage"
        exit 1
    fi
fi

echo -e "${GREEN}Setting up labels for repository: ${REPO_NAME}${NC}"

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}Error: GitHub CLI (gh) is not installed${NC}"
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check if authenticated
if ! gh auth status &>/dev/null; then
    echo -e "${RED}Error: Not authenticated with GitHub CLI${NC}"
    echo "Run: gh auth login"
    exit 1
fi

# Get labels file path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LABELS_FILE="${SCRIPT_DIR}/../.github/labels.json"

if [ ! -f "$LABELS_FILE" ]; then
    echo -e "${RED}Error: Labels file not found: ${LABELS_FILE}${NC}"
    exit 1
fi

echo -e "${YELLOW}Reading labels from: ${LABELS_FILE}${NC}"

# Read labels from JSON file and create them
LABELS_COUNT=0

# Check for jq or python3 for JSON parsing
if command -v jq &> /dev/null; then
    # Use jq to parse JSON
    while IFS= read -r label_json; do
        NAME=$(echo "$label_json" | jq -r '.name')
        COLOR=$(echo "$label_json" | jq -r '.color')
        DESCRIPTION=$(echo "$label_json" | jq -r '.description // ""')
        
        if [ -z "$NAME" ] || [ -z "$COLOR" ] || [ "$NAME" = "null" ] || [ "$COLOR" = "null" ]; then
            echo -e "${YELLOW}Warning: Skipping invalid label entry${NC}"
            continue
        fi
        
        # Check if label already exists
        if gh label list --repo "$REPO_NAME" --json name --jq ".[] | select(.name == \"$NAME\") | .name" 2>/dev/null | grep -q "^${NAME}$"; then
            echo -e "${YELLOW}Label '${NAME}' already exists, updating...${NC}"
            if [ -n "$DESCRIPTION" ] && [ "$DESCRIPTION" != "null" ]; then
                gh label edit "$NAME" --repo "$REPO_NAME" --color "$COLOR" --description "$DESCRIPTION" || true
            else
                gh label edit "$NAME" --repo "$REPO_NAME" --color "$COLOR" || true
            fi
        else
            echo -e "${GREEN}Creating label: ${NAME}${NC}"
            if [ -n "$DESCRIPTION" ] && [ "$DESCRIPTION" != "null" ]; then
                gh label create "$NAME" --repo "$REPO_NAME" --color "$COLOR" --description "$DESCRIPTION" || true
            else
                gh label create "$NAME" --repo "$REPO_NAME" --color "$COLOR" || true
            fi
        fi
        
        LABELS_COUNT=$((LABELS_COUNT + 1))
    done < <(jq -c '.[]' "$LABELS_FILE")
elif command -v python3 &> /dev/null; then
    # Use python3 to parse JSON
    python3 << EOF
import json
import subprocess
import sys

with open("$LABELS_FILE", 'r') as f:
    labels = json.load(f)

count = 0
for label in labels:
    name = label.get('name', '')
    color = label.get('color', '')
    description = label.get('description', '')
    
    if not name or not color:
        print("Warning: Skipping invalid label entry", file=sys.stderr)
        continue
    
    # Check if label exists
    result = subprocess.run(
        ['gh', 'label', 'list', '--repo', '$REPO_NAME', '--json', 'name', '--jq', f'.[] | select(.name == "{name}") | .name'],
        capture_output=True,
        text=True
    )
    
    exists = name in result.stdout
    
    if exists:
        print(f"Label '{name}' already exists, updating...")
        cmd = ['gh', 'label', 'edit', name, '--repo', '$REPO_NAME', '--color', color]
        if description:
            cmd.extend(['--description', description])
        subprocess.run(cmd, capture_output=True)
    else:
        print(f"Creating label: {name}")
        cmd = ['gh', 'label', 'create', name, '--repo', '$REPO_NAME', '--color', color]
        if description:
            cmd.extend(['--description', description])
        subprocess.run(cmd, capture_output=True)
    
    count += 1

print(f"Processed {count} labels")
EOF
    LABELS_COUNT=$(python3 -c "import json; print(len(json.load(open('$LABELS_FILE'))))")
else
    echo -e "${RED}Error: Neither jq nor python3 is available for JSON parsing${NC}"
    echo "Please install jq (https://stedolan.github.io/jq/) or ensure python3 is available"
    exit 1
fi

echo -e "${GREEN}✓ Successfully processed ${LABELS_COUNT} labels${NC}"
echo -e "${GREEN}Labels are now available in repository: ${REPO_NAME}${NC}"
