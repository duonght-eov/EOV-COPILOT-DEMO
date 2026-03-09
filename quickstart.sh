#!/bin/bash
# SpeedMaint Intelligence - Quick Start Script

echo "======================================"
echo "SpeedMaint Intelligence - Quick Start"
echo "======================================"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "\n${YELLOW}1. Setting up API Gateway...${NC}"
cd api-gateway

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "   Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "   Installing dependencies..."
pip install -q -r requirements.txt

# Copy .env if not exists
if [ ! -f ".env" ]; then
    echo "   Creating .env from example..."
    cp .env.example .env
fi

# Create data directory
mkdir -p data/uploads

echo -e "${GREEN}   ✓ API Gateway ready${NC}"

cd ..

echo -e "\n${YELLOW}2. Starting services...${NC}"
echo "   $ cd api-gateway && uvicorn app.main:app --reload --port 8080"

echo -e "\n${YELLOW}3. Access points:${NC}"
echo "   - API Gateway: http://localhost:8080"
echo "   - Swagger UI:  http://localhost:8080/docs"

echo -e "\n${GREEN}Setup complete!${NC}"
