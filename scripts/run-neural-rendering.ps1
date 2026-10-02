[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$BenillaArgs
)

$ErrorActionPreference = 'Stop'

if (-not $env:DLSS_SDK) {
    throw 'DLSS_SDK must point at an NVIDIA NGX SDK containing nvsdk_ngx_d.lib.'
}

if (-not $env:WOW_DLSSNR_DLL) {
    $defaultRuntime = Join-Path $PSScriptRoot '..\target\release\dlssnr\Ada Lovelace+\nvngx_dlssnr.dll'
    if (Test-Path -LiteralPath $defaultRuntime -PathType Leaf) {
        # Prefer the checked project-local runtime over a stale SDK-directory variable. The SDK
        # import library is for linking; it is not the proprietary Feature-18 runtime.
        $env:WOW_DLSSNR_DLL = $defaultRuntime
    } else {
        $sdkRuntime = Join-Path $env:DLSS_SDK 'nvngx_dlssnr.dll'
        if (Test-Path -LiteralPath $sdkRuntime -PathType Leaf) {
            # NVIDIA's SDK distribution puts the Feature-18 runtime at its root, not below the
            # architecture-named directory the client uses for an installed runtime bundle.
            $env:WOW_DLSSNR_DLL = $sdkRuntime
        } elseif (-not $env:WOW_DLSSNR_DIR) {
            throw "Set WOW_DLSSNR_DLL or WOW_DLSSNR_DIR, or place the RTX 40 runtime at $defaultRuntime."
        }
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
# PowerShell's call operator resolves `.dll` through file associations, although this is an EXE
# image with a deliberate NGX-required filename. `UseShellExecute = $false` calls CreateProcess
# directly, which accepts the PE image without consulting that association table.
$start = [System.Diagnostics.ProcessStartInfo]::new()
$start.FileName = $neuralExecutable
$start.UseShellExecute = $false
$start.Arguments = if ($null -eq $BenillaArgs) { '' } else { [string]::Join(' ', $BenillaArgs) }
$process = [System.Diagnostics.Process]::Start($start)
$process.WaitForExit()
exit $process.ExitCode
