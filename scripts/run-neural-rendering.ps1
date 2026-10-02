[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BenillaArgs
)

$ErrorActionPreference = 'Stop'

if (-not $env:DLSS_SDK) {
    throw 'DLSS_SDK must point at an NVIDIA NGX SDK containing nvsdk_ngx_d.lib.'
}

if (-not $env:WOW_DLSSNR_DLL -and -not $env:WOW_DLSSNR_DIR) {
    $defaultRuntime = Join-Path $PSScriptRoot '..\target\release\dlssnr\Ada Lovelace+\nvngx_dlssnr.dll'
    if (-not (Test-Path -LiteralPath $defaultRuntime -PathType Leaf)) {
        throw "Set WOW_DLSSNR_DLL or WOW_DLSSNR_DIR, or place the RTX 40 runtime at $defaultRuntime."
    }
}

cargo build -p benilla --release --features dlss
if ($LASTEXITCODE -ne 0) {
    throw "cargo build failed with exit code $LASTEXITCODE"
}

$release = Join-Path $PSScriptRoot '..\target\release'
$source = Join-Path $release 'benilla.exe'
$neuralExecutable = Join-Path $release 'nvngx.dll'
Copy-Item -LiteralPath $source -Destination $neuralExecutable -Force

# NGX checks the image filename, not a loaded module. Keep benilla.exe untouched for normal runs.
& $neuralExecutable @BenillaArgs
exit $LASTEXITCODE
