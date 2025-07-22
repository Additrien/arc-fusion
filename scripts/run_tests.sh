#!/bin/bash

# Arc-Fusion Test Runner Script
# Run tests for the document processing pipeline

set -e  # Exit on error

echo "=== Arc-Fusion Test Runner ==="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    print_status "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
print_status "Activating virtual environment..."
source venv/bin/activate

# Install/upgrade dependencies
print_status "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create logs directory for tests
mkdir -p logs

# Set environment variables for tests
export ENVIRONMENT=test
export GOOGLE_API_KEY=test_key
export WEAVIATE_URL=http://localhost:8080
export ENABLE_FILE_LOGGING=false

# Parse command line arguments
TEST_TYPE="all"
COVERAGE=true
VERBOSE=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --unit)
            TEST_TYPE="unit"
            shift
            ;;
        --integration)
            TEST_TYPE="integration"
            shift
            ;;
        --api)
            TEST_TYPE="api"
            shift
            ;;
        --no-coverage)
            COVERAGE=false
            shift
            ;;
        --quiet)
            VERBOSE=false
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --unit         Run only unit tests"
            echo "  --integration  Run only integration tests"
            echo "  --api          Run only API tests"
            echo "  --no-coverage  Skip coverage reporting"
            echo "  --quiet        Run with minimal output"
            echo "  --help         Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build pytest command
PYTEST_CMD="pytest"

# Add test selection based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD -m unit"
        print_status "Running unit tests..."
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD -m integration"
        print_status "Running integration tests..."
        ;;
    api)
        PYTEST_CMD="$PYTEST_CMD -m api"
        print_status "Running API tests..."
        ;;
    *)
        print_status "Running all tests..."
        ;;
esac

# Add coverage if enabled
if [ "$COVERAGE" = true ]; then
    PYTEST_CMD="$PYTEST_CMD --cov=app --cov-report=html --cov-report=term"
fi

# Add verbosity
if [ "$VERBOSE" = false ]; then
    PYTEST_CMD="$PYTEST_CMD -q"
fi

# Run the tests
print_status "Executing: $PYTEST_CMD"
echo ""

if $PYTEST_CMD; then
    print_status "Tests completed successfully!"
    
    if [ "$COVERAGE" = true ]; then
        print_status "Coverage report generated in htmlcov/"
        echo ""
        echo "Open htmlcov/index.html to view detailed coverage report"
    fi
    
    exit 0
else
    print_error "Tests failed!"
    exit 1
fi 