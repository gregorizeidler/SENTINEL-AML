#!/bin/bash

# AML-FT Adversarial Simulation Demo Runner
# This script sets up and runs the complete demonstration

echo "🎯 AML-FT ADVERSARIAL SIMULATION DEMO"
echo "===================================="
echo

# Check Python version
python_version=$(python3 --version 2>&1)
echo "✅ Python version: $python_version"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📚 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  Warning: .env file not found"
    echo "📝 Creating .env from example..."
    cp env.example .env
    echo "✅ Please edit .env with your API keys before running the demo"
    echo "   Example: OPENAI_API_KEY=your_key_here"
    echo
    read -p "Press Enter to continue once you've added your API keys..."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data/generated
mkdir -p demo_results

# Run the demo
echo
echo "🚀 Starting AML-FT Adversarial Simulation..."
echo "============================================"
echo

python demo.py

echo
echo "🎉 Demo completed!"
echo "📊 Results saved in demo_results/ directory"
echo "📝 Check the generated files for detailed analysis"
echo

# Deactivate virtual environment
deactivate 