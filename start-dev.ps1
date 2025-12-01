# Quick Start Script for EXAID Development
# Run this script to start both backend and frontend servers

Write-Host "🚀 Starting EXAID Development Environment" -ForegroundColor Cyan
Write-Host ""

# Get the script's directory as project root (works across different machines)
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Start FastAPI backend in a new terminal (assumes virtual environment is activated or python is in PATH)
Write-Host "📡 Starting FastAPI Backend (port 8000)..." -ForegroundColor Green
$backendJob = Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; Write-Host '🔧 FastAPI Backend Server' -ForegroundColor Cyan; python -m uvicorn demos.backend.server:app --reload" -PassThru

Start-Sleep -Seconds 2

# Start Next.js frontend in a new terminal
Write-Host "⚛️  Starting Next.js Frontend (port 3000)..." -ForegroundColor Green
$frontendJob = Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$projectRoot\demos\frontend'; Write-Host '⚛️  Next.js Frontend Server' -ForegroundColor Cyan; npm run dev" -PassThru

Write-Host ""
Write-Host "✅ Both servers are starting!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Access the application:" -ForegroundColor Cyan
Write-Host "   • Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "   • Backend:  http://localhost:8000" -ForegroundColor White
Write-Host ""
Write-Host "💡 To stop both servers, close both terminal windows" -ForegroundColor Yellow
Write-Host "   or press Ctrl+C in each terminal" -ForegroundColor Yellow
Write-Host ""
