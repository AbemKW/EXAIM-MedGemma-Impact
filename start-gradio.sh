#!/bin/bash
# Quick start script for EXAIM Gradio Demo (Unix/Mac/Linux)

echo "🏥 EXAIM Gradio Demo - Quick Start"
echo "================================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if requirements are installed
echo "📚 Installing/updating dependencies..."
pip install -r requirements-gradio.txt --quiet

# Check for .env file
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  Warning: .env file not found!"
    echo "Please create a .env file from .env.example and add your API keys."
    echo ""
    
    read -p "Would you like to create .env now? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp .env.example .env
        echo "✅ Created .env file. Please edit it and add your API keys."
        echo "Opening .env in default editor..."
        ${EDITOR:-nano} .env
        echo ""
        echo "After saving your API keys, press any key to continue..."
        read -n 1 -s
    else
        echo ""
        echo "❌ Cannot proceed without API keys. Please create .env file and try again."
        exit 1
    fi
fi

echo ""
echo "🚀 Starting EXAIM Gradio Demo..."
echo ""
echo "📍 The demo will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the server"
echo ""

# Run the Gradio app
python app_gradio.py
