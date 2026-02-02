# Quick start script for EXAIM Gradio Demo
# This script sets up and runs the Gradio interface

Write-Host "🏥 EXAIM Gradio Demo - Quick Start" -ForegroundColor Cyan
Write-Host "=================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv")) {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
}

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# Check if requirements are installed
Write-Host "📚 Installing/updating dependencies..." -ForegroundColor Yellow
pip install -r requirements-gradio.txt --quiet

# Check for .env file
if (-not (Test-Path ".env")) {
    Write-Host ""
    Write-Host "⚠️  Warning: .env file not found!" -ForegroundColor Red
    Write-Host "Please create a .env file from .env.example and add your API keys." -ForegroundColor Yellow
    Write-Host ""
    
    $createEnv = Read-Host "Would you like to create .env now? (y/n)"
    if ($createEnv -eq "y") {
        Copy-Item .env.example .env
        Write-Host "✅ Created .env file. Please edit it and add your API keys." -ForegroundColor Green
        Write-Host "Press any key to open .env in notepad..." -ForegroundColor Yellow
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        notepad .env
        Write-Host ""
        Write-Host "After saving your API keys, press any key to continue..." -ForegroundColor Yellow
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    } else {
        Write-Host ""
        Write-Host "❌ Cannot proceed without API keys. Please create .env file and try again." -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "🚀 Starting EXAIM Gradio Demo..." -ForegroundColor Green
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Run the Gradio app
python app_gradio.py
