# Restart script for EXAID backend and frontend servers
# This script kills existing processes and restarts them

Write-Host "🛑 Stopping EXAID servers..." -ForegroundColor Yellow

# Kill backend process (Python/FastAPI on port 8000)
Write-Host "Stopping backend server..." -ForegroundColor Yellow

# Method 1: Kill by port 8000
$backendProcesses = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
if ($backendProcesses) {
    foreach ($pid in $backendProcesses) {
        $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
        if ($proc) {
            Write-Host "Killing backend process on port 8000 (PID: $pid, Name: $($proc.ProcessName))..." -ForegroundColor Red
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
} else {
    Write-Host "No backend process found on port 8000" -ForegroundColor Gray
}

# Method 2: Kill Python processes that are running server.py (more aggressive)
$pythonProcesses = Get-Process -Name "python","pythonw","python3" -ErrorAction SilentlyContinue
foreach ($pythonProc in $pythonProcesses) {
    $shouldKill = $false
    $reason = ""
    
    try {
        # Check command line for "server.py" or "backend"
        $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($pythonProc.Id)").CommandLine
        if ($commandLine) {
            if ($commandLine -like "*server.py*" -or $commandLine -like "*backend*server*" -or $commandLine -like "*demos\backend*") {
                $shouldKill = $true
                $reason = "Backend server.py process detected"
            }
        }
    } catch {
        # If we can't check command line, check ports
    }
    
    # Also check if process is using backend port
    if (-not $shouldKill) {
        try {
            $ports = Get-NetTCPConnection -OwningProcess $pythonProc.Id -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LocalPort
            if ($ports -contains 8000) {
                $shouldKill = $true
                $reason = "Using backend port 8000"
            }
        } catch {
            # Ignore errors
        }
    }
    
    if ($shouldKill) {
        Write-Host "Killing Python backend process (PID: $($pythonProc.Id)) - $reason..." -ForegroundColor Red
        Stop-Process -Id $pythonProc.Id -Force -ErrorAction SilentlyContinue
    }
}

# Kill frontend processes more aggressively
Write-Host "Stopping frontend server..." -ForegroundColor Yellow

# Method 1: Kill by port (3000 and 3001)
$frontendPorts = @(3000, 3001)
foreach ($port in $frontendPorts) {
    $frontendProcesses = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    if ($frontendProcesses) {
        foreach ($pid in $frontendProcesses) {
            $proc = Get-Process -Id $pid -ErrorAction SilentlyContinue
            if ($proc) {
                Write-Host "Killing process on port $port (PID: $pid, Name: $($proc.ProcessName))..." -ForegroundColor Red
                Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
            }
        }
    }
}

# Method 2: Kill all Node.js processes that might be Next.js (more aggressive)
# Find Node processes and check their command line or ports
$nodeProcesses = Get-Process -Name "node" -ErrorAction SilentlyContinue
foreach ($nodeProc in $nodeProcesses) {
    $shouldKill = $false
    $reason = ""
    
    try {
        # Check command line for "next" or "frontend"
        $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $($nodeProc.Id)").CommandLine
        if ($commandLine) {
            if ($commandLine -like "*next*" -or $commandLine -like "*frontend*" -or $commandLine -like "*demos\frontend*") {
                $shouldKill = $true
                $reason = "Next.js process detected"
            }
        }
    } catch {
        # If we can't check command line, check ports
    }
    
    # Also check if process is using frontend ports
    if (-not $shouldKill) {
        try {
            $ports = Get-NetTCPConnection -OwningProcess $nodeProc.Id -ErrorAction SilentlyContinue | Select-Object -ExpandProperty LocalPort
            if ($ports -contains 3000 -or $ports -contains 3001) {
                $shouldKill = $true
                $reason = "Using frontend port"
            }
        } catch {
            # Ignore errors
        }
    }
    
    if ($shouldKill) {
        Write-Host "Killing Node.js process (PID: $($nodeProc.Id)) - $reason..." -ForegroundColor Red
        Stop-Process -Id $nodeProc.Id -Force -ErrorAction SilentlyContinue
    }
}

# Remove Next.js lock file and .next directory if needed
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$lockFile = Join-Path $scriptDir "demos\frontend\.next\dev\lock"
$nextDir = Join-Path $scriptDir "demos\frontend\.next"

if (Test-Path $lockFile) {
    Write-Host "Removing Next.js lock file..." -ForegroundColor Yellow
    Remove-Item -Path $lockFile -Force -ErrorAction SilentlyContinue
}

# Also try to remove the entire .next directory if it exists (more aggressive)
# This ensures a clean restart
if (Test-Path $nextDir) {
    Write-Host "Cleaning Next.js cache directory..." -ForegroundColor Yellow
    try {
        Remove-Item -Path $nextDir -Recurse -Force -ErrorAction SilentlyContinue
        Write-Host "Next.js cache cleared" -ForegroundColor Green
    } catch {
        Write-Host "Could not fully clear cache (some files may be locked)" -ForegroundColor Yellow
    }
}

# Wait a moment for processes to fully terminate
Write-Host "Waiting for processes to terminate..." -ForegroundColor Gray
Start-Sleep -Seconds 3

# Double-check ports are free
Write-Host "Verifying ports are free..." -ForegroundColor Gray

# Check backend port 8000
$checkPort8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($checkPort8000) {
    Write-Host "Warning: Port 8000 may still be in use" -ForegroundColor Yellow
    $pids = $checkPort8000 | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($pid in $pids) {
        Write-Host "Force killing remaining process on port 8000 (PID: $pid)..." -ForegroundColor Red
        Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Seconds 1
}

# Check frontend ports 3000/3001
$checkPort3000 = Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue
$checkPort3001 = Get-NetTCPConnection -LocalPort 3001 -ErrorAction SilentlyContinue
if ($checkPort3000 -or $checkPort3001) {
    Write-Host "Warning: Ports 3000/3001 may still be in use" -ForegroundColor Yellow
    # Try one more time to kill processes on these ports
    if ($checkPort3000) {
        $pids = $checkPort3000 | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($pid in $pids) {
            Write-Host "Force killing remaining process on port 3000 (PID: $pid)..." -ForegroundColor Red
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
    if ($checkPort3001) {
        $pids = $checkPort3001 | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($pid in $pids) {
            Write-Host "Force killing remaining process on port 3001 (PID: $pid)..." -ForegroundColor Red
            Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 2
}

Write-Host "🚀 Starting EXAID servers..." -ForegroundColor Green

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = $scriptDir

# Start backend in a new window
Write-Host "Starting backend server..." -ForegroundColor Cyan
$backendScript = Join-Path $projectRoot "demos\backend\server.py"

# Use python -u for unbuffered output to ensure clean startup
# Clear PYTHONPATH to avoid any cached modules
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$projectRoot'; `$env:PYTHONUNBUFFERED='1'; python -u `"$backendScript`"" -WindowStyle Normal

# Wait a moment for backend to start and verify it's running
Write-Host "Waiting for backend to initialize..." -ForegroundColor Gray
Start-Sleep -Seconds 4

# Verify backend started successfully
$backendCheck = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
if ($backendCheck) {
    Write-Host "Backend started successfully on port 8000" -ForegroundColor Green
} else {
    Write-Host "Warning: Backend may not have started on port 8000" -ForegroundColor Yellow
}

# Start frontend in a new window
Write-Host "Starting frontend server..." -ForegroundColor Cyan
$frontendDir = Join-Path $projectRoot "demos\frontend"
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$frontendDir'; npm run dev" -WindowStyle Normal

Write-Host "✅ Servers restarted!" -ForegroundColor Green
Write-Host "Backend: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan

