# Quick Start Script for EXAID Development
# Run this script to start both backend and frontend servers and automatically open the browser

Write-Host "🚀 Starting EXAID Development Environment" -ForegroundColor Cyan
Write-Host ""

# Get the script's directory as project root (works across different machines)
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

# Function to check if a server is responding via HTTP
function Test-ServerReady {
    param([string]$Url, [int]$TimeoutSeconds = 60)
    
    $startTime = Get-Date
    $endTime = $startTime.AddSeconds($TimeoutSeconds)
    
    while ((Get-Date) -lt $endTime) {
        try {
            $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 2 -UseBasicParsing -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                return $true
            }
        } catch {
            # Server not ready yet, continue waiting
        }
        Start-Sleep -Milliseconds 1000
    }
    return $false
}

# Start FastAPI backend in a new terminal (assumes virtual environment is activated or python is in PATH)
Write-Host "📡 Starting FastAPI Backend (port 8000)..." -ForegroundColor Green
$backendJob = Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; Write-Host '🔧 FastAPI Backend Server' -ForegroundColor Cyan; .\.venv\Scripts\python.exe -m uvicorn demos.backend.server:app --reload" -PassThru

Start-Sleep -Seconds 2

# Start Next.js frontend in a new terminal
Write-Host "⚛️  Starting Next.js Frontend (port 3000)..." -ForegroundColor Green
$frontendJob = Start-Process pwsh -ArgumentList "-NoExit", "-Command", "cd '$projectRoot\demos\frontend'; Write-Host '⚛️  Next.js Frontend Server' -ForegroundColor Cyan; npm run dev" -PassThru

Write-Host ""
Write-Host "⏳ Waiting for servers to be ready..." -ForegroundColor Yellow

# Wait for frontend to be ready (Next.js typically takes 10-20 seconds)
Write-Host "   Checking frontend (http://localhost:3000)..." -ForegroundColor Gray
if (Test-ServerReady -Url "http://localhost:3000" -TimeoutSeconds 60) {
    Write-Host "   ✅ Frontend is ready!" -ForegroundColor Green
} else {
    Write-Host "   ⚠️  Frontend may not be ready yet, but opening browser anyway..." -ForegroundColor Yellow
    Write-Host "   (The page will load automatically once the server is ready)" -ForegroundColor Gray
}

Write-Host ""
Write-Host "✅ Both servers are starting!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Access the application:" -ForegroundColor Cyan
Write-Host "   • Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "   • Backend:  http://localhost:8000" -ForegroundColor White
Write-Host ""

# Automatically open the browser
$frontendUrl = "http://localhost:3000"
Write-Host "🌐 Opening browser..." -ForegroundColor Cyan
try {
    Start-Process $frontendUrl
    Write-Host "   ✅ Browser opened successfully!" -ForegroundColor Green
} catch {
    Write-Host "   ⚠️  Could not open browser automatically. Please navigate to: $frontendUrl" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "💡 To stop both servers, close both terminal windows" -ForegroundColor Yellow
Write-Host "   or press Ctrl+C in each terminal" -ForegroundColor Yellow
Write-Host ""
