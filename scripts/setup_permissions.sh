#!/bin/bash

# Setup script to ensure proper permissions for Docker volumes
# This script should be run before docker compose up

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up permissions for Arc-Fusion Docker volumes...${NC}"

# Get current user info
USER_ID=$(id -u)
GROUP_ID=$(id -g)
USERNAME=$(whoami)

echo -e "${YELLOW}Current user: ${USERNAME} (UID=${USER_ID}, GID=${GROUP_ID})${NC}"

# Create directories if they don't exist
mkdir -p logs data

# Set permissions
echo -e "${YELLOW}Setting up directory permissions...${NC}"
chmod 755 logs data
chown ${USER_ID}:${GROUP_ID} logs data 2>/dev/null || {
    echo -e "${RED}Warning: Could not change ownership. You may need to run:${NC}"
    echo -e "${RED}  sudo chown -R ${USER_ID}:${GROUP_ID} logs data${NC}"
}

# Create .env file with user info if it doesn't exist
if [[ ! -f .env.docker ]]; then
    echo -e "${YELLOW}Creating .env.docker with user permissions...${NC}"
    cat > .env.docker << EOF
# Docker user permissions
USER_ID=${USER_ID}
GROUP_ID=${GROUP_ID}
EOF
else
    # Update existing .env.docker
    echo -e "${YELLOW}Updating .env.docker with current user permissions...${NC}"
    grep -v "^USER_ID=" .env.docker > .env.docker.tmp || true
    grep -v "^GROUP_ID=" .env.docker.tmp > .env.docker || true
    echo "USER_ID=${USER_ID}" >> .env.docker
    echo "GROUP_ID=${GROUP_ID}" >> .env.docker
    rm -f .env.docker.tmp
fi

echo -e "${GREEN}Permissions setup complete!${NC}"
echo -e "${YELLOW}You can now run: docker compose --env-file .env.docker up --build${NC}" 