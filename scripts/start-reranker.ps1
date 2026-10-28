param(
    [int]$Port = 11436,
    [string]$Model = "qllama/bge-reranker-v2-m3:latest",
    [switch]$Foreground
)

$ErrorActionPreference = "Stop"

function Test-RerankerReady {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$Port/health" -TimeoutSec 2
        return $health.status -eq "ok"
    } catch {
        return $false
    }
}

if (Test-RerankerReady) {
    Write-Host "Reranker is already ready on port $Port."
    exit 0
}

$lib = Join-Path $env:LOCALAPPDATA "Programs\Ollama\lib\ollama"
$server = Join-Path $lib "llama-server.exe"
if (-not (Test-Path -LiteralPath $server)) {
    throw "llama-server.exe not found: $server"
}

$modelfile = ollama show $Model --modelfile
$fromLine = $modelfile | Select-String '^FROM ' | Select-Object -Last 1
if (-not $fromLine) {
    throw "Cannot resolve GGUF blob from model: $Model"
}
$blob = $fromLine.Line.Substring(5).Trim()
if (-not (Test-Path -LiteralPath $blob)) {
    throw "GGUF blob not found: $blob"
}

$env:PATH = "$lib;$lib\cuda_v12;$env:PATH"
$arguments = @(
    "--model", $blob,
    "--host", "0.0.0.0",
    "--port", "$Port",
    "--rerank",
    "--pooling", "rank",
    "--ctx-size", "8192",
    "--alias", ($Model -replace ':latest$', ''),
    "--no-webui",
    "--offline"
)

if ($Foreground) {
    & $server @arguments
    exit $LASTEXITCODE
}

$runtime = Join-Path $PSScriptRoot "..\data\logs"
New-Item -ItemType Directory -Force -Path $runtime | Out-Null
$stdout = Join-Path $runtime "reranker-$Port.out.log"
$stderr = Join-Path $runtime "reranker-$Port.err.log"
$process = Start-Process -FilePath $server -ArgumentList $arguments -WindowStyle Hidden -PassThru `
    -RedirectStandardOutput $stdout -RedirectStandardError $stderr

$ready = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    Start-Sleep -Seconds 1
    if ($process.HasExited) {
        break
    }
    if (Test-RerankerReady) {
        $ready = $true
        break
    }
}
if (-not $ready) {
    throw "Reranker failed to start. Check $stderr"
}

Write-Host "Reranker started on http://0.0.0.0:$Port (PID $($process.Id))."
Write-Host "Model: $Model"
Write-Host "Endpoint: http://127.0.0.1:$Port/rerank"

