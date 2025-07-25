#!/bin/bash

# Wrapper script for Arc-Fusion Docker deployment with automatic permission setup

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Arc-Fusion Docker Setup${NC}"

# Run permission setup
echo -e "${YELLOW}Step 1: Setting up permissions...${NC}"
./scripts/setup_permissions.sh

# Build and start containers
echo -e "${YELLOW}Step 2: Building and starting containers...${NC}"

# Use both .env files if they exist
ENV_FILES=""
if [[ -f .env ]]; then
    ENV_FILES="--env-file .env"
fi
if [[ -f .env.docker ]]; then
    ENV_FILES="${ENV_FILES} --env-file .env.docker"
fi

# Start containers
if [[ "$1" == "build" ]]; then
    echo -e "${YELLOW}Force rebuilding containers...${NC}"
    docker compose ${ENV_FILES} down
    docker compose ${ENV_FILES} up --build -d
elif [[ "$1" == "down" ]]; then
    echo -e "${YELLOW}Stopping containers...${NC}"
    docker compose ${ENV_FILES} down
    exit 0
else
    docker compose ${ENV_FILES} up -d
fi

echo -e "${GREEN}✅ Arc-Fusion is starting up!${NC}"
echo -e "${YELLOW}Health check: http://localhost:8000/health${NC}"
echo -e "${YELLOW}API docs: http://localhost:8000/docs${NC}"

# Wait a moment and check health
echo -e "${YELLOW}Waiting for services to be ready...${NC}"
sleep 10

if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Services are healthy and ready!${NC}"
else
    echo -e "${YELLOW}⚠️  Services are still starting up. Check logs with: docker compose logs -f${NC}"
fi