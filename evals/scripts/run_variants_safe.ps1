param(
    [Parameter(Mandatory=$true)]
    [string]$OutputDir,
    [Parameter(Mandatory=$true)]
    [string]$EvalRunId
)

# Collect any extra positional args passed after the named parameters
$remaining = $args

# Normalize path for comparison (handle / and \\ and case-insensitive)
$norm = ($OutputDir -replace '/','\\').TrimEnd('\')
$norm = $norm.ToLower()

$forbidden = @('evals/data/runs','evals/data/runs_baseline_gemini25flashlite')

foreach ($f in $forbidden) {
    $fNorm = ($f -replace '/','\\').ToLower()
    if ($norm -eq $fNorm -or $norm.EndsWith('\' + $fNorm) -or $norm.EndsWith($fNorm)) {
        Write-Error "Refusing to write to '$OutputDir' — this path is protected. Pick a different output directory."
        exit 1
    }
}

# Build argument list and display the command being executed
$arguments = @('-m','evals.cli.run_variants','--output',$OutputDir,'--eval-run-id',$EvalRunId) + $remaining
Write-Host "Executing: python $($arguments -join ' ')"

# Run the python command and propagate its exit code
$proc = Start-Process -FilePath 'python' -ArgumentList $arguments -NoNewWindow -Wait -PassThru
exit $proc.ExitCode
