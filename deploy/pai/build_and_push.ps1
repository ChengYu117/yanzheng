$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$ignoreFile = Join-Path $scriptDir ".dockerignore"

if (-not $env:ACR_REGISTRY) { throw "Set ACR_REGISTRY, e.g. registry-vpc.cn-hangzhou.aliyuncs.com" }
if (-not $env:ACR_NAMESPACE) { throw "Set ACR_NAMESPACE" }
if (-not $env:ACR_REPOSITORY) { throw "Set ACR_REPOSITORY" }

$imageTag = if ($env:IMAGE_TAG) { $env:IMAGE_TAG } else { Get-Date -Format "yyyyMMdd-HHmmss" }
$imageUri = "$($env:ACR_REGISTRY)/$($env:ACR_NAMESPACE)/$($env:ACR_REPOSITORY):$imageTag"

$tempRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("sae-re-pai-" + [guid]::NewGuid().ToString("N"))
$contextDir = Join-Path $tempRoot "repo"
New-Item -ItemType Directory -Path $contextDir -Force | Out-Null

try {
    Push-Location $repoRoot
    tar.exe --exclude-from="$ignoreFile" -cf "$tempRoot\\context.tar" .
    Pop-Location
    Push-Location $contextDir
    tar.exe -xf "$tempRoot\\context.tar"
    Pop-Location

    Write-Host "Building image: $imageUri"
    docker build -f "$contextDir\\deploy\\pai\\Dockerfile" -t $imageUri $contextDir

    if ($env:ACR_USERNAME -and $env:ACR_PASSWORD) {
        $env:ACR_PASSWORD | docker login $env:ACR_REGISTRY --username $env:ACR_USERNAME --password-stdin
    } else {
        Write-Host "ACR_USERNAME / ACR_PASSWORD not set. Please run docker login $($env:ACR_REGISTRY) manually if needed."
    }

    docker push $imageUri
    Write-Host "Pushed image: $imageUri"
}
finally {
    if (Test-Path $tempRoot) {
        Remove-Item -Recurse -Force $tempRoot
    }
}
