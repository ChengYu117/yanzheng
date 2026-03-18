param(
    [string]$InputDir = "",
    [string]$OutputDir = "",
    [string]$ApiKey = "",
    [string]$BaseUrl = "",
    [string]$Model = "",
    [string]$PythonExe = "",
    [int]$TopLatents = 20,
    [int]$TopN = 10,
    [int]$ControlN = 5,
    [string]$Groups = "G1,G5,G20",
    [double]$Temperature = 0.0,
    [int]$MaxRetries = 3,
    [int]$RequestTimeout = 0,
    [switch]$DryRunPrompts
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Resolve-Path (Join-Path $scriptDir "..\..")
$envFile = Join-Path $scriptDir "bailian_qwen35plus.env"
$exampleFile = Join-Path $scriptDir "bailian_qwen35plus.env.example"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^\s*#' -or $_ -match '^\s*$') { return }
        $pair = $_ -split '=', 2
        if ($pair.Length -eq 2) {
            [Environment]::SetEnvironmentVariable($pair[0], $pair[1], "Process")
        }
    }
} elseif (Test-Path $exampleFile) {
    Write-Host "No local env file found at $envFile. Falling back to process env / script parameters."
}

if (-not $InputDir) {
    $InputDir = $env:JUDGE_INPUT_DIR
}
if (-not $InputDir) {
    $InputDir = "outputs/sae_eval_full_max"
}

if (-not $OutputDir) {
    if ($env:JUDGE_OUTPUT_DIR) {
        $OutputDir = $env:JUDGE_OUTPUT_DIR
    } else {
        $OutputDir = Join-Path $InputDir "ai_judge_qwen35plus"
    }
}

if ($ApiKey) {
    $env:OPENAI_API_KEY = $ApiKey
}
if ($BaseUrl) {
    $env:OPENAI_BASE_URL = $BaseUrl
}
if ($Model) {
    $env:OPENAI_MODEL = $Model
}

if (-not $env:OPENAI_BASE_URL) {
    $env:OPENAI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
}
if (-not $env:OPENAI_MODEL) {
    $env:OPENAI_MODEL = "qwen3.5-plus"
}
if (-not $env:OPENAI_EXTRA_BODY_JSON) {
    $env:OPENAI_EXTRA_BODY_JSON = '{"enable_thinking":false}'
}
if ($RequestTimeout -gt 0) {
    $env:OPENAI_HTTP_TIMEOUT = "$RequestTimeout"
}

if (-not $env:OPENAI_API_KEY -and -not $DryRunPrompts) {
    throw "OPENAI_API_KEY is missing. Pass -ApiKey or create deploy/local/bailian_qwen35plus.env."
}

$bundleManifest = Join-Path $InputDir "judge_bundle\manifest.json"
$directManifest = Join-Path $InputDir "manifest.json"
if (-not (Test-Path $bundleManifest) -and -not (Test-Path $directManifest)) {
    throw "judge_bundle is missing under '$InputDir'. Re-run run_sae_evaluation.py with the current codebase so it exports judge_bundle/ first."
}

$pythonCandidates = @()
if ($PythonExe) {
    $pythonCandidates += $PythonExe
}
if ($env:PYTHON_EXE) {
    $pythonCandidates += $env:PYTHON_EXE
}
if ($env:VIRTUAL_ENV) {
    $pythonCandidates += (Join-Path $env:VIRTUAL_ENV "Scripts\python.exe")
}
$pythonCandidates += (Join-Path $projectRoot ".venv\Scripts\python.exe")
$pythonCandidates += "python"

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq "python") {
        $pythonExe = $candidate
        break
    }
    if (Test-Path $candidate) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    throw "Could not resolve a usable Python interpreter."
}

$argsList = @(
    (Join-Path $projectRoot "run_ai_re_judge.py"),
    "--input-dir", $InputDir,
    "--output-dir", $OutputDir,
    "--model", $env:OPENAI_MODEL,
    "--top-latents", "$TopLatents",
    "--top-n", "$TopN",
    "--control-n", "$ControlN",
    "--groups", $Groups,
    "--temperature", "$Temperature",
    "--max-retries", "$MaxRetries"
)
if ($RequestTimeout -gt 0) {
    $argsList += @("--request-timeout", "$RequestTimeout")
}

if ($DryRunPrompts) {
    $argsList += "--dry-run-prompts"
}

Push-Location $projectRoot
try {
    & $pythonExe @argsList
} finally {
    Pop-Location
}
