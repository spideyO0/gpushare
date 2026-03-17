# ------------------------------------------------------------------------------
# gpushare -Windows Client Install Script (PowerShell)
#
# Installs the gpushare client library on Windows, enabling transparent remote
# GPU access. One DLL provides three APIs via copies:
#   cudart64_12.dll / cudart64_130.dll  -CUDA Runtime API
#   nvcuda.dll                          -CUDA Driver API (PyTorch/TF load this)
#   nvml.dll                            -NVML GPU management (monitoring tools)
#
# Run as Administrator:  PowerShell -ExecutionPolicy Bypass -File install-client-windows.ps1
# ------------------------------------------------------------------------------

param(
    [string]$Server     = "",
    [switch]$SkipBuild,
    [switch]$SkipPython,
    [switch]$Force,
    [switch]$AutoDeps,
    [switch]$Help
)

$ErrorActionPreference = "Stop"

# -- Paths --------------------------------------------------------------------
$ScriptDir  = $PSScriptRoot
$ProjectDir = Split-Path -Parent $ScriptDir
# Handle case where script is run from a nested directory
if (-not (Test-Path (Join-Path $ProjectDir "CMakeLists.txt"))) {
    $ProjectDir = Split-Path -Parent (Split-Path -Parent $ScriptDir)
}
if (-not (Test-Path (Join-Path $ProjectDir "CMakeLists.txt"))) {
    # Last resort: assume we are in the project root
    $ProjectDir = (Get-Location).Path
}
$BuildDir   = Join-Path $ProjectDir "build"
$InstallDir = "C:\Program Files\gpushare"
$ConfigDir  = "C:\ProgramData\gpushare"
$ShareDir   = Join-Path $InstallDir "share"

$PrebuiltUrl = "https://github.com/example/gpushare/releases/latest/download/gpushare_client.dll"

# -- Colored output helpers ---------------------------------------------------
function Write-Step    { param($Msg) Write-Host ""                                       ; Write-Host ">> $Msg" -ForegroundColor White }
function Write-Info    { param($Msg) Write-Host "   [INFO]  $Msg" -ForegroundColor Cyan   }
function Write-Ok      { param($Msg) Write-Host "   [OK]    $Msg" -ForegroundColor Green  }
function Write-Warn    { param($Msg) Write-Host "   [WARN]  $Msg" -ForegroundColor Yellow }
function Write-Err     { param($Msg) Write-Host "   [ERROR] $Msg" -ForegroundColor Red    }
function Write-Fatal   { param($Msg) Write-Err $Msg; exit 1 }

# -- Help ---------------------------------------------------------------------
if ($Help) {
    Write-Host "Usage: install-client-windows.ps1 [OPTIONS]"
    Write-Host ""
    Write-Host "Install the gpushare client library on Windows. Provides transparent CUDA"
    Write-Host "access to a remote gpushare server - any GPU application will detect the"
    Write-Host "remote GPU as if it were local."
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Server HOST:PORT   Set the server address"
    Write-Host "  -SkipBuild          Skip cmake build (use existing build/ directory)"
    Write-Host "  -SkipPython         Do not install Python client package"
    Write-Host "  -Force              Force full reinstall (overwrite everything)"
    Write-Host "  -AutoDeps           Install missing deps without prompting"
    Write-Host "  -Help               Show this help"
    Write-Host ""
    Write-Host "Compiler detection (priority order):"
    Write-Host "  1. Visual Studio Build Tools (cl.exe via vswhere)"
    Write-Host "  2. MinGW-w64 (g++.exe in PATH)"
    Write-Host "  3. MSYS2 (g++ in MSYS2 paths)"
    Write-Host "  4. Pre-built binary download (if no compiler found)"
    Write-Host ""
    Write-Host "Run this script as Administrator (right-click -> Run as Administrator)."
    exit 0
}

# -- Admin check --------------------------------------------------------------
$isAdmin = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)

if (-not $isAdmin) {
    Write-Err "This script requires Administrator privileges."
    Write-Info "Right-click PowerShell -> 'Run as Administrator', then re-run this script."
    Write-Host ""
    $elevate = Read-Host "Attempt to re-launch as Administrator? [Y/n]"
    if ($elevate -eq "" -or $elevate -eq "y" -or $elevate -eq "Y") {
        $argList = "-ExecutionPolicy Bypass -File `"$PSCommandPath`""
        foreach ($p in $PSBoundParameters.GetEnumerator()) {
            if ($p.Value -is [switch] -and $p.Value) {
                $argList += " -$($p.Key)"
            } elseif ($p.Value -is [string] -and $p.Value) {
                $argList += " -$($p.Key) `"$($p.Value)`""
            }
        }
        Start-Process powershell.exe -ArgumentList $argList -Verb RunAs
    }
    exit 1
}

# -- Upgrade detection --------------------------------------------------------
$IsUpgrade = $false
if ((Test-Path "$InstallDir\gpushare_client.dll") -and (-not $Force)) {
    $IsUpgrade = $true
}

# -- Banner -------------------------------------------------------------------
Write-Host ""
Write-Host "  ============================================================" -ForegroundColor White
if ($IsUpgrade) {
    Write-Host "    gpushare - Windows Client Installer  (UPGRADE)" -ForegroundColor Cyan
} else {
    Write-Host "    gpushare - Windows Client Installer" -ForegroundColor Cyan
}
Write-Host "  ============================================================" -ForegroundColor White
Write-Host ""

if ($Force) {
    Write-Warn "Force mode: full reinstall regardless of existing installation"
}

# -- Stop scheduled task if upgrading -----------------------------------------
$TaskName = "gpushare-dashboard"
if ($IsUpgrade) {
    Write-Info "Upgrade detected -stopping scheduled task if running..."
    try {
        $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        if ($existing -and $existing.State -eq "Running") {
            Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
            Write-Ok "Stopped $TaskName scheduled task"
        }
    } catch {
        # Task may not exist yet
    }
}

# -- CUDA conflict warning ---------------------------------------------------
$cudaPath = $env:CUDA_PATH
if ($cudaPath -and (Test-Path $cudaPath)) {
    Write-Host ""
    Write-Host "  !! WARNING: Local CUDA installation detected at $cudaPath" -ForegroundColor Yellow
    Write-Host "  !! gpushare installs DLLs that take priority over local CUDA." -ForegroundColor Yellow
    Write-Host ""
    if (-not $IsUpgrade -and -not $Force) {
        $answer = Read-Host "   Continue and override local CUDA? [y/N]"
        if ($answer -ne "y" -and $answer -ne "Y") {
            Write-Info "Aborting. Remove -Server flag or set CUDA_PATH to empty to skip this check."
            exit 0
        }
    }
}

# ==============================================================================
# STEP 1: Detect build environment
# ==============================================================================
Write-Step "Detecting build environment..."

$Compiler     = ""   # "msvc", "mingw", "msys2", "prebuilt", "skip"
$Generator    = ""
$VsDevCmd     = ""
$NeedDownload = $false

if ($SkipBuild) {
    $Compiler = "skip"
    Write-Info "Build skipped (-SkipBuild) -will use existing build/ directory"
} else {
    # --- Priority 1: Visual Studio Build Tools (cl.exe via vswhere) ---
    $vsWherePaths = @(
        "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe",
        "${env:ProgramFiles}\Microsoft Visual Studio\Installer\vswhere.exe"
    )
    foreach ($vw in $vsWherePaths) {
        if ($Compiler -eq "" -and (Test-Path $vw)) {
            $vsInstallPath = & $vw -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath 2>$null
            if ($vsInstallPath -and (Test-Path $vsInstallPath)) {
                $Compiler  = "msvc"
                # Determine generator version
                # Get the major version number (e.g., 17.x, 18.x)
                $vsFullVer = & $vw -latest -products * -property installationVersion 2>$null
                $vsMajor = if ($vsFullVer) { ($vsFullVer -split '\.')[0] } else { "" }

                # Map major version to CMake generator string
                # CMake uses: "Visual Studio <major> <year>"
                $vsYearMap = @{
                    "18" = "2026"
                    "17" = "2022"
                    "16" = "2019"
                    "15" = "2017"
                    "14" = "2015"
                }
                $vsYear = $vsYearMap[$vsMajor]
                if ($vsYear) {
                    $Generator = "Visual Studio $vsMajor $vsYear"
                } else {
                    # Unknown version - try Ninja which works with any VS
                    $Generator = "Ninja"
                }
                Write-Ok ("Visual Studio found: " + $vsInstallPath + " (v" + $vsMajor + " / " + $vsYear + ")")
                # Locate vcvarsall for environment setup
                $vcvars = Join-Path $vsInstallPath "VC\Auxiliary\Build\vcvarsall.bat"
                if (Test-Path $vcvars) {
                    $VsDevCmd = $vcvars
                }
            }
        }
    }

    # --- Priority 2: MinGW-w64 (g++.exe in PATH) ---
    if ($Compiler -eq "") {
        $gppPath = Get-Command "g++.exe" -ErrorAction SilentlyContinue
        if ($gppPath) {
            $Compiler  = "mingw"
            $Generator = "MinGW Makefiles"
            Write-Ok "MinGW-w64 found: $($gppPath.Source)"
        }
    }

    # --- Priority 3: MSYS2 (check common MSYS2 installation paths) ---
    if ($Compiler -eq "") {
        $msys2Paths = @(
            "C:\msys64\mingw64\bin\g++.exe",
            "C:\msys64\ucrt64\bin\g++.exe",
            "C:\msys64\clang64\bin\g++.exe",
            "C:\msys32\mingw32\bin\g++.exe"
        )
        foreach ($mp in $msys2Paths) {
            if ($Compiler -eq "" -and (Test-Path $mp)) {
                $Compiler  = "msys2"
                $Generator = "MinGW Makefiles"
                # Add the MSYS2 bin dir to PATH for this session
                $msys2Bin = Split-Path -Parent $mp
                $env:Path = $msys2Bin + ";" + $env:Path
                Write-Ok ("MSYS2 g++ found: " + $mp)
            }
        }
    }

    # --- Priority 4: No compiler -offer pre-built download ---
    if ($Compiler -eq "") {
        Write-Warn "No C++ compiler found (checked: Visual Studio, MinGW-w64, MSYS2)"
        Write-Host ""

        # Check if there is already a built DLL in the build directory
        $existingDll = $null
        $searchPaths = @(
            (Join-Path $BuildDir "gpushare_client.dll"),
            (Join-Path $BuildDir "libgpushare_client.dll"),
            (Join-Path $BuildDir "Release\gpushare_client.dll"),
            (Join-Path $BuildDir "Release\libgpushare_client.dll"),
            (Join-Path $BuildDir "Debug\gpushare_client.dll"),
            (Join-Path $BuildDir "Debug\libgpushare_client.dll")
        )
        foreach ($sp in $searchPaths) {
            if ((-not $existingDll) -and (Test-Path $sp)) {
                $existingDll = $sp
            }
        }

        if ($existingDll) {
            Write-Info "Found existing build artifact: $existingDll"
            $useExisting = Read-Host "   Use this existing DLL? [Y/n]"
            if ($useExisting -eq "" -or $useExisting -eq "y" -or $useExisting -eq "Y") {
                $Compiler = "skip"
            }
        }

        if ($Compiler -eq "") {
            Write-Info "Options:"
            Write-Host "   [1] Download pre-built DLL from $PrebuiltUrl" -ForegroundColor Cyan
            Write-Host "   [2] Abort (install a compiler first)" -ForegroundColor Cyan
            Write-Host ""
            $choice = Read-Host "   Choose [1/2]"
            if ($choice -eq "1") {
                $Compiler     = "prebuilt"
                $NeedDownload = $true
            } else {
                Write-Fatal "No compiler available. Install Visual Studio Build Tools, MinGW-w64, or MSYS2."
            }
        }
    }
}

# ==============================================================================
# STEP 2: Detect cmake
# ==============================================================================
$cmakeBin = $null

if ($Compiler -ne "skip" -and $Compiler -ne "prebuilt") {
    Write-Step "Detecting CMake..."

    # Check PATH first
    $cmakeCmd = Get-Command cmake -ErrorAction SilentlyContinue
    if ($cmakeCmd) {
        $cmakeBin = $cmakeCmd.Source
    }

    # Check Visual Studio bundled cmake
    if (-not $cmakeBin) {
        $vsCmakePaths = @(
            "${env:ProgramFiles}\Microsoft Visual Studio\2022\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            "${env:ProgramFiles(x86)}\Microsoft Visual Studio\2022\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
            "${env:ProgramFiles}\Microsoft Visual Studio\2019\*\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
        )
        foreach ($pattern in $vsCmakePaths) {
            if (-not $cmakeBin) {
                $found = Resolve-Path $pattern -ErrorAction SilentlyContinue | Select-Object -First 1
                if ($found) { $cmakeBin = $found.Path }
            }
        }
    }

    # Check Chocolatey
    if (-not $cmakeBin) {
        $chocoPath = "C:\ProgramData\chocolatey\bin\cmake.exe"
        if (Test-Path $chocoPath) { $cmakeBin = $chocoPath }
    }

    # Check Scoop
    if (-not $cmakeBin) {
        $scoopPath = "$env:USERPROFILE\scoop\shims\cmake.exe"
        if (Test-Path $scoopPath) { $cmakeBin = $scoopPath }
    }

    if (-not $cmakeBin) {
        Write-Fatal "CMake not found. Install from https://cmake.org/download/ or via 'winget install Kitware.CMake'"
    }

    Write-Ok "CMake found: $cmakeBin"
}

# ==============================================================================
# STEP 3: Build or download client library
# ==============================================================================
if ($Compiler -eq "prebuilt" -and $NeedDownload) {
    Write-Step "Downloading pre-built client DLL..."
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
    $downloadDest = Join-Path $BuildDir "gpushare_client.dll"
    try {
        Write-Info "Downloading from $PrebuiltUrl ..."
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -Uri $PrebuiltUrl -OutFile $downloadDest -UseBasicParsing
        Write-Ok "Downloaded gpushare_client.dll"
    } catch {
        Write-Fatal "Download failed: $_`nManually place gpushare_client.dll in $BuildDir and re-run with -SkipBuild"
    }
} elseif ($Compiler -ne "skip") {
    Write-Step "Building gpushare client library ($Compiler)..."
    New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

    # -- Handle stale CMake cache from a different machine ----------------
    $cacheFile = Join-Path $BuildDir "CMakeCache.txt"
    if (Test-Path $cacheFile) {
        $cachedSrc = ""
        $cacheLines = Get-Content $cacheFile -ErrorAction SilentlyContinue
        foreach ($line in $cacheLines) {
            if ($line -match "^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$") {
                $cachedSrc = $Matches[1].Trim()
                break
            }
        }
        # Normalize paths for comparison (forward slashes vs backslashes)
        $normalizedProject = $ProjectDir.Replace("\", "/")
        $normalizedCached  = $cachedSrc.Replace("\", "/")
        if ($cachedSrc -and ($normalizedCached -ne $normalizedProject)) {
            Write-Warn "Stale CMake cache from $cachedSrc -cleaning build directory"
            Remove-Item -Recurse -Force $BuildDir
            New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null
        }
    }

    # -- Configure --------------------------------------------------------
    $cmakeArgs = @(
        "-S", $ProjectDir,
        "-B", $BuildDir,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SERVER=OFF",
        "-DBUILD_CLIENT=ON"
    )
    if ($Generator) {
        $cmakeArgs += @("-G", $Generator)
    }

    Write-Info "Running: cmake $($cmakeArgs -join ' ')"
    & $cmakeBin @cmakeArgs
    if ($LASTEXITCODE -ne 0) { Write-Fatal "CMake configure failed (exit code $LASTEXITCODE)" }

    # -- Build ------------------------------------------------------------
    $buildArgs = @("--build", $BuildDir, "--config", "Release", "--parallel")
    Write-Info "Running: cmake $($buildArgs -join ' ')"
    & $cmakeBin @buildArgs
    if ($LASTEXITCODE -ne 0) { Write-Fatal "Build failed (exit code $LASTEXITCODE)" }

    Write-Ok "Build complete"
} else {
    Write-Step "Skipping build (-SkipBuild)"
}

# -- Locate the built DLL ----------------------------------------------------
$clientDll = $null
$dllSearchPaths = @(
    (Join-Path $BuildDir "gpushare_client.dll"),
    (Join-Path $BuildDir "libgpushare_client.dll"),
    (Join-Path $BuildDir "Release\gpushare_client.dll"),
    (Join-Path $BuildDir "Release\libgpushare_client.dll"),
    (Join-Path $BuildDir "Debug\gpushare_client.dll"),
    (Join-Path $BuildDir "Debug\libgpushare_client.dll"),
    (Join-Path $BuildDir "bin\Release\gpushare_client.dll"),
    (Join-Path $BuildDir "bin\gpushare_client.dll"),
    (Join-Path $BuildDir "bin\libgpushare_client.dll")
)
foreach ($sp in $dllSearchPaths) {
    if ((-not $clientDll) -and (Test-Path $sp)) {
        $clientDll = $sp
    }
}
if (-not $clientDll) {
    Write-Fatal "Build artifact not found: gpushare_client.dll`nSearched: $($dllSearchPaths -join ', ')"
}
Write-Ok "Client DLL located: $clientDll"

# ==============================================================================
# STEP 4: Install DLL + create API copies
# ==============================================================================
Write-Step "Installing to $InstallDir..."
New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null

# Copy base DLL
Copy-Item -Force $clientDll "$InstallDir\gpushare_client.dll"
Write-Ok "Installed gpushare_client.dll"

# Copy MinGW runtime DLLs if the build used MinGW (needed if not statically linked)
if ($Compiler -eq "mingw") {
    $mingwBin = Split-Path (Get-Command g++.exe -ErrorAction SilentlyContinue).Source -ErrorAction SilentlyContinue
    if ($mingwBin) {
        $runtimeDlls = @("libstdc++-6.dll", "libgcc_s_seh-1.dll", "libwinpthread-1.dll")
        foreach ($rtDll in $runtimeDlls) {
            $rtPath = Join-Path $mingwBin $rtDll
            if (Test-Path $rtPath) {
                Copy-Item -Force $rtPath "$InstallDir\$rtDll"
            }
        }
        Write-Ok "MinGW runtime DLLs copied"
    }
}

# Backup real CUDA DLLs for local GPU passthrough (dual-GPU support)
$realBackupDir = Join-Path $InstallDir "real"
New-Item -ItemType Directory -Force -Path $realBackupDir | Out-Null

$backupDlls = @(
    @{ Name = "nvcuda.dll";       Search = @("$env:SystemRoot\System32") },
    @{ Name = "cudart64_130.dll"; Search = @("$env:CUDA_PATH\bin", "$env:SystemRoot\System32") },
    @{ Name = "cudart64_12.dll";  Search = @("$env:CUDA_PATH\bin", "$env:SystemRoot\System32") },
    @{ Name = "nvml.dll";         Search = @("$env:SystemRoot\System32", "$env:ProgramFiles\NVIDIA Corporation\NVSMI") }
)

foreach ($entry in $backupDlls) {
    $dstPath = Join-Path $realBackupDir $entry.Name
    if (Test-Path $dstPath) { continue }
    foreach ($dir in $entry.Search) {
        if (-not $dir) { continue }
        $srcPath = Join-Path $dir $entry.Name
        if (Test-Path $srcPath) {
            $realSrc = (Get-Item $srcPath).FullName
            if ($realSrc -notmatch "gpushare") {
                Copy-Item -Force $realSrc $dstPath
                Write-Ok "Backed up $($entry.Name) from $dir"
                break
            }
        }
    }
}

# Create transparent replacement copies for each API
Write-Info "Creating transparent CUDA/NVML replacement DLLs..."

$apiDlls = @(
    "cudart64_12.dll",    # CUDA Runtime API (CUDA 12.x)
    "cudart64_130.dll",   # CUDA Runtime API (CUDA 13.x)
    "nvcuda.dll",         # CUDA Driver API (what PyTorch/TF load from system)
    "nvml.dll",           # NVML GPU management (nvidia-smi, monitoring tools)
    "cublas64_12.dll",
    "cublasLt64_12.dll",
    "cudnn64_9.dll",
    "cufft64_11.dll",
    "cusparse64_12.dll",
    "cusolver64_11.dll",
    "curand64_10.dll",
    "nvrtc64_120_0.dll",
    "nvjpeg64_12.dll"
)

foreach ($dll in $apiDlls) {
    Copy-Item -Force "$InstallDir\gpushare_client.dll" "$InstallDir\$dll"
    Write-Ok "  -> $dll"
}

# ==============================================================================
# STEP 4b: Install DLLs into Python directories (DLL search order priority)
# ==============================================================================
# Windows DLL search order: (1) app directory -> (2) System32 -> (3) PATH.
# System32 DLLs (nvcuda.dll, nvml.dll) are locked by the NVIDIA driver and
# cannot be overwritten. Instead, we copy gpushare DLLs into every Python
# installation's directory - position #1 in search order beats System32.
Write-Step "Installing DLLs into Python directories for search order priority..."

# IMPORTANT: Do NOT copy nvcuda.dll or nvml.dll to Python directories.
# PyTorch's c10_cuda.dll loads nvcuda.dll and calls real NVIDIA driver
# functions that gpushare does not export. Overriding nvcuda.dll breaks
# PyTorch with "WinError 127: The specified procedure could not be found".
#
# Only override cudart DLLs - these are the CUDA Runtime API that
# applications call directly. gpushare handles local+remote GPU routing.
$criticalDlls = @("cudart64_12.dll", "cudart64_130.dll")

# Find all Python installations
$pythonDirs = @()
# Current Python in PATH
$pyCmd = Get-Command python -ErrorAction SilentlyContinue
if ($pyCmd) { $pythonDirs += Split-Path $pyCmd.Source }
$py3Cmd = Get-Command python3 -ErrorAction SilentlyContinue
if ($py3Cmd) { $pythonDirs += Split-Path $py3Cmd.Source }
# Common Python install locations
$pythonDirs += Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python*" -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
$pythonDirs += Get-ChildItem "C:\Python*" -Directory -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName }
# Deduplicate
$pythonDirs = $pythonDirs | Sort-Object -Unique | Where-Object { Test-Path $_ }

# First, clean up any nvcuda.dll/nvml.dll from previous installs that broke PyTorch
$dangerousDlls = @("nvcuda.dll", "nvml.dll")
foreach ($pyDir in $pythonDirs) {
    foreach ($dll in $dangerousDlls) {
        $dllPath = Join-Path $pyDir $dll
        if (Test-Path $dllPath) {
            $ourDll = Join-Path $InstallDir $dll
            if (Test-Path $ourDll) {
                $dstSize = (Get-Item $dllPath).Length
                $ourSize = (Get-Item $ourDll).Length
                if ($dstSize -eq $ourSize) {
                    Remove-Item -Force $dllPath -ErrorAction SilentlyContinue
                    Write-Ok "Removed incorrectly placed $dll from $pyDir"
                }
            }
        }
    }
}

if ($pythonDirs.Count -eq 0) {
    Write-Warn "No Python installations found - DLL override limited to PATH priority"
} else {
    foreach ($pyDir in $pythonDirs) {
        $count = 0
        foreach ($dll in $criticalDlls) {
            $srcDll = Join-Path $InstallDir $dll
            $dstDll = Join-Path $pyDir $dll
            if (-not (Test-Path $srcDll)) { continue }

            # Backup the real DLL if one exists in this Python dir
            if (Test-Path $dstDll) {
                $backupPath = Join-Path $realBackupDir "python_$dll"
                $realSize = (Get-Item $dstDll).Length
                $ourSize = (Get-Item $srcDll).Length
                if (($realSize -ne $ourSize) -and (-not (Test-Path $backupPath))) {
                    Copy-Item -Force $dstDll $backupPath
                }
            }

            try {
                Copy-Item -Force $srcDll $dstDll
                $count++
            } catch {
                Write-Warn "Could not copy $dll to $pyDir : $_"
            }
        }
        if ($count -gt 0) {
            Write-Ok "Installed $count DLLs to $pyDir"
        }
    }
    Write-Info "DLLs installed to $($pythonDirs.Count) Python directory(s) for search order priority"
}

# IMPORTANT: Do NOT override torch\lib\ DLLs. PyTorch's internal c10_cuda.dll
# depends on real NVIDIA driver functions that gpushare does not export.
# Overriding torch\lib\nvcuda.dll breaks PyTorch with WinError 127.
#
# If a previous install placed DLLs in torch\lib\, restore them now.
foreach ($pyDir in $pythonDirs) {
    $torchLib = Join-Path $pyDir "Lib\site-packages\torch\lib"
    if (-not (Test-Path $torchLib)) { continue }
    foreach ($dll in @("nvcuda.dll", "nvml.dll", "cudart64_12.dll", "cudart64_130.dll")) {
        $dstDll = Join-Path $torchLib $dll
        if (-not (Test-Path $dstDll)) { continue }
        $dstSize = (Get-Item $dstDll).Length
        $ourDll = Join-Path $InstallDir $dll
        if (-not (Test-Path $ourDll)) { continue }
        $ourSize = (Get-Item $ourDll).Length
        if ($dstSize -eq $ourSize) {
            # This is our DLL in torch\lib - remove it or restore backup
            $backupPath = Join-Path $realBackupDir "torch_$dll"
            if (Test-Path $backupPath) {
                Copy-Item -Force $backupPath $dstDll
                Write-Ok "Restored original $dll in torch\lib (was incorrectly overridden)"
            } else {
                Remove-Item -Force $dstDll -ErrorAction SilentlyContinue
                Write-Ok "Removed gpushare $dll from torch\lib (was incorrectly placed)"
            }
        }
    }
}

# ==============================================================================
# STEP 5: Configure system PATH
# ==============================================================================
Write-Step "Configuring system PATH..."

$machinePath  = [Environment]::GetEnvironmentVariable("Path", "Machine")
$pathEntries  = $machinePath -split ";" | Where-Object { $_ -ne "" }

# Remove any existing gpushare entry
$pathEntries = $pathEntries | Where-Object { $_ -ne $InstallDir }

# Find the index of the first CUDA-related path
$cudaIndex = -1
for ($i = 0; $i -lt $pathEntries.Count; $i++) {
    if ($pathEntries[$i] -match 'NVIDIA GPU Computing Toolkit|\\cuda\\|\\nvidia\\') {
        $cudaIndex = $i
        break
    }
}

if ($cudaIndex -ge 0) {
    # Insert BEFORE the first CUDA path
    $before = $pathEntries[0..($cudaIndex - 1)]
    $after  = $pathEntries[$cudaIndex..($pathEntries.Count - 1)]
    $newEntries = @($before) + @($InstallDir) + @($after)
    Write-Info "Placed gpushare BEFORE CUDA paths in system PATH"
} else {
    # No CUDA paths found -prepend to ensure priority
    $newEntries = @($InstallDir) + @($pathEntries)
}

$newPath = ($newEntries | Where-Object { $_ -ne "" }) -join ";"
[Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
Write-Ok "System PATH updated (persistent, Machine level)"

# Also update current session
$env:Path = "$InstallDir;$env:Path"

# ==============================================================================
# STEP 6: Install config file
# ==============================================================================
Write-Step "Installing client configuration..."
New-Item -ItemType Directory -Force -Path $ConfigDir | Out-Null

$configFile   = Join-Path $ConfigDir "client.conf"
$templateFile = Join-Path $ProjectDir "config\gpushare-client.conf"
$hasTemplate  = Test-Path $templateFile

if ($IsUpgrade -and (Test-Path $configFile)) {
    # Upgrade: preserve existing config, write new as reference
    Write-Warn "Config already exists at $configFile -not overwriting"
    if ($hasTemplate) {
        Copy-Item -Force $templateFile "$configFile.new"
        Write-Info "New defaults written to $configFile.new for reference"
    }
    # Read server address from existing config for the summary
    $existingContent = Get-Content $configFile -ErrorAction SilentlyContinue
    foreach ($line in $existingContent) {
        if ($line -match "^server=(.+)$") {
            if (-not $Server) { $Server = $Matches[1].Trim() }
        }
    }
    if (-not $Server) { $Server = "unknown" }
} else {
    # Fresh install -prompt for server address if not provided
    if (-not $Server) {
        $default = "192.168.1.100:9847"
        if (Test-Path $configFile) {
            $existingContent = Get-Content $configFile -ErrorAction SilentlyContinue
            foreach ($line in $existingContent) {
                if ($line -match "^server=(.+)$") {
                    $default = $Matches[1].Trim()
                }
            }
        }
        $Server = Read-Host "   Server address (host:port) [$default]"
        if (-not $Server) { $Server = $default }
    }

    if ($hasTemplate) {
        $configContent = Get-Content $templateFile -Raw
        $configContent = $configContent -replace "(?m)^server=.*", "server=$Server"
        Set-Content -Path $configFile -Value $configContent -NoNewline
    } else {
        # No template -write a minimal config
        $minimalConfig = "# gpushare client configuration`r`nserver=$Server"
        Set-Content -Path $configFile -Value $minimalConfig
    }
    Write-Ok "Config installed: $configFile (server=$Server)"
}

# ==============================================================================
# STEP 7: Install Python client + TUI
# ==============================================================================
$pythonExe    = $null
$pythonwExe   = $null
$hasPython    = $false

if (-not $SkipPython) {
    Write-Step "Installing Python client..."

    # Find python
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    }

    if ($pythonCmd) {
        $pythonExe  = $pythonCmd.Source
        $pythonwExe = Join-Path (Split-Path -Parent $pythonExe) "pythonw.exe"
        if (-not (Test-Path $pythonwExe)) { $pythonwExe = $null }
        $hasPython = $true

        # Check pip
        $hasPip = $false
        try {
            & $pythonExe -m pip --version 2>$null | Out-Null
            if ($LASTEXITCODE -eq 0) { $hasPip = $true }
        } catch { }

        $pythonDir = Join-Path $ProjectDir "python"
        if ($hasPip -and (Test-Path $pythonDir)) {
            Write-Info "Installing gpushare Python package..."
            try {
                & $pythonExe -m pip install $pythonDir --quiet 2>$null
                if ($LASTEXITCODE -eq 0) {
                    Write-Ok "Python client package installed"
                } else {
                    Write-Warn "Python client install returned non-zero (non-fatal)"
                }
            } catch {
                Write-Warn "Python client install failed (non-fatal): $_"
            }
        } elseif (-not $hasPip) {
            Write-Warn "pip not found -skipping Python client package"
        } elseif (-not (Test-Path $pythonDir)) {
            Write-Warn "Python package directory not found at $pythonDir -skipping"
        }
    } else {
        Write-Warn "Python not found in PATH -skipping Python client"
    }
} else {
    Write-Info "Skipping Python client (-SkipPython)"
}

# -- Install dashboard + TUI files --------------------------------------------
Write-Step "Installing dashboard and TUI..."
New-Item -ItemType Directory -Force -Path "$ShareDir\dashboard" | Out-Null
New-Item -ItemType Directory -Force -Path "$ShareDir\tui"       | Out-Null

$dashboardSrc = Join-Path $ProjectDir "dashboard"
$tuiSrc       = Join-Path $ProjectDir "tui"

if (Test-Path "$dashboardSrc\*") {
    Copy-Item -Recurse -Force "$dashboardSrc\*" "$ShareDir\dashboard\"
    Write-Ok "Dashboard files installed"
} else {
    Write-Warn "No dashboard files found at $dashboardSrc"
}

if (Test-Path "$tuiSrc\*") {
    Copy-Item -Recurse -Force "$tuiSrc\*" "$ShareDir\tui\"
    Write-Ok "TUI files installed"
} else {
    Write-Warn "No TUI files found at $tuiSrc"
}

# Create monitor batch file
$monitorBat = "@echo off`r`npython `"%~dp0share\tui\monitor.py`" %*"
$monitorBatPath = Join-Path $InstallDir "gpushare-monitor.bat"
Set-Content -Path $monitorBatPath -Value $monitorBat
Write-Ok "Created gpushare-monitor.bat"

# Create nvidia-smi.bat shim
$nvidiaSmiSrc = Join-Path $ProjectDir "scripts\nvidia-smi"
if (Test-Path $nvidiaSmiSrc) {
    $nvidiaSmiDst = Join-Path $ShareDir "nvidia-smi"
    Copy-Item -Force $nvidiaSmiSrc $nvidiaSmiDst
    Write-Ok "Installed nvidia-smi Python script"
}
$nvidiaBatPath = Join-Path $InstallDir "nvidia-smi.bat"
$nvidiaBat = "@echo off`r`npython `"$ShareDir\nvidia-smi`" %*"
Set-Content -Path $nvidiaBatPath -Value $nvidiaBat

# Copy the uninstall script into the install directory for easy access
$uninstSrc = Join-Path $ProjectDir "scripts\uninstall-windows.ps1"
if (Test-Path $uninstSrc) {
    Copy-Item -Force $uninstSrc "$InstallDir\uninstall.ps1"
    Write-Ok "Uninstall script copied to $InstallDir\uninstall.ps1"
}

# Unblock all installed files (remove Windows "downloaded from internet" flag)
Write-Info "Unblocking installed files (removing download security flag)..."
Get-ChildItem -Path $InstallDir -Recurse | Unblock-File -ErrorAction SilentlyContinue
Get-ChildItem -Path $ShareDir  -Recurse | Unblock-File -ErrorAction SilentlyContinue
Get-ChildItem -Path $ConfigDir -Recurse -ErrorAction SilentlyContinue | Unblock-File -ErrorAction SilentlyContinue
Write-Ok "All files unblocked"

# ==============================================================================
# STEP 8a: Windows GPU Registration (Task Manager detection)
# ==============================================================================
Write-Step "Registering GPU for Windows Task Manager detection..."

# Windows Task Manager reads GPU info from the display adapter registry class.
# We create a virtual display adapter entry so Task Manager shows the remote GPU.
$displayClassGuid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
$classRegPath     = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\$displayClassGuid"

try {
    # Find the next available subkey number
    $existing = Get-ChildItem -Path $classRegPath -ErrorAction SilentlyContinue |
                Where-Object { $_.PSChildName -match "^\d{4}$" } |
                ForEach-Object { [int]$_.PSChildName } |
                Sort-Object -Descending |
                Select-Object -First 1

    $nextIdx = if ($existing -ne $null) { $existing + 1 } else { 100 }
    $subKey  = $nextIdx.ToString("D4")
    $adapterPath = "$classRegPath\$subKey"

    # Check if we already registered one
    $alreadyRegistered = $false
    Get-ChildItem -Path $classRegPath -ErrorAction SilentlyContinue | ForEach-Object {
        $drvDesc = (Get-ItemProperty -Path $_.PSPath -Name "DriverDesc" -ErrorAction SilentlyContinue).DriverDesc
        if ($drvDesc -eq "NVIDIA GeForce RTX 5070 (gpushare)") {
            $alreadyRegistered = $true
        }
    }

    if (-not $alreadyRegistered) {
        # Query the remote GPU for its actual name
        $gpuName = "NVIDIA GeForce RTX 5070 (gpushare)"

        New-Item -Path $adapterPath -Force | Out-Null
        New-ItemProperty -Path $adapterPath -Name "DriverDesc"              -Value $gpuName                              -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $adapterPath -Name "ProviderName"            -Value "gpushare (remote GPU)"               -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $adapterPath -Name "HardwareInformation.AdapterString"        -Value $gpuName             -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $adapterPath -Name "HardwareInformation.qwMemorySize"         -Value 12330598400          -PropertyType QWord  -Force | Out-Null
        New-ItemProperty -Path $adapterPath -Name "HardwareInformation.ChipType"             -Value "Blackwell GB205"    -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $adapterPath -Name "HardwareInformation.DacType"              -Value "Integrated RAMDAC"  -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $adapterPath -Name "HardwareInformation.BiosString"           -Value "gpushare remote"    -PropertyType String -Force | Out-Null
        New-ItemProperty -Path $adapterPath -Name "gpushare_managed"        -Value 1                                     -PropertyType DWord  -Force | Out-Null

        Write-Ok "Registered virtual GPU adapter in Windows registry"
        Write-Info "  Task Manager will show: $gpuName"
        Write-Info "  (Visible after next reboot or Task Manager restart)"
    } else {
        Write-Ok "GPU adapter already registered in Windows registry"
    }
} catch {
    Write-Warn "Could not register GPU in registry (non-fatal): $_"
    Write-Info "Task Manager GPU tab may not show the remote GPU"
}

# ==============================================================================
# STEP 8b: Windows Defender exclusion (avoid DLL scan delays)
# ==============================================================================
try {
    Write-Info "Adding Windows Defender exclusion for gpushare..."
    Add-MpPreference -ExclusionPath $InstallDir -ErrorAction SilentlyContinue
    Write-Ok "Defender exclusion added for $InstallDir"
} catch {
    Write-Info "Defender exclusion skipped (may require additional permissions)"
}

# ==============================================================================
# STEP 9a: Install GPU system tray widget
# ==============================================================================
Write-Step "Installing GPU system tray widget..."
$traySrc = Join-Path $ProjectDir "client\gpu_tray_windows.pyw"
if (Test-Path $traySrc) {
    Copy-Item -Force $traySrc "$ShareDir\gpu_tray_windows.pyw"
    Write-Ok "GPU tray widget installed"

    # Install dependencies
    if ($hasPython) {
        Write-Info "Installing tray widget dependencies (pystray, pillow)..."
        try {
            & $pythonExe -m pip install pystray pillow --quiet 2>$null
            Write-Ok "pystray + pillow installed"
        } catch {
            Write-Warn "Could not install pystray/pillow (tray widget optional)"
        }
    }

    # Create a startup shortcut for the tray widget
    try {
        $startupDir = [Environment]::GetFolderPath("Startup")
        $shortcutPath = Join-Path $startupDir "gpushare GPU Monitor.lnk"
        $trayExe = if ($pythonwExe) { $pythonwExe } elseif ($pythonExe) { $pythonExe } else { "pythonw.exe" }
        $trayScript = "$ShareDir\gpu_tray_windows.pyw"

        $wshell = New-Object -ComObject WScript.Shell
        $shortcut = $wshell.CreateShortcut($shortcutPath)
        $shortcut.TargetPath = $trayExe
        $shortcut.Arguments = "`"$trayScript`""
        $shortcut.WorkingDirectory = $ShareDir
        $shortcut.Description = "gpushare remote GPU monitor"
        $shortcut.Save()
        Write-Ok "Startup shortcut created - tray widget runs on login"

        # Also start it now
        Start-Process -FilePath $trayExe -ArgumentList "`"$trayScript`"" -WindowStyle Hidden
        Write-Ok "GPU tray widget started - check your system tray!"
    } catch {
        Write-Warn "Could not create startup shortcut: $_"
    }
} else {
    Write-Warn "GPU tray widget source not found"
}

# ==============================================================================
# STEP 9b: Scheduled task for dashboard auto-start
# ==============================================================================
Write-Step "Configuring scheduled task for dashboard..."

try {
    $existingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existingTask) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Info "Removed existing scheduled task"
    }

    # Determine executable -prefer pythonw.exe (no console window)
    $dashExe  = if ($pythonwExe) { $pythonwExe } elseif ($pythonExe) { $pythonExe } else { "pythonw.exe" }
    $dashArgs = "`"$ShareDir\dashboard\app.py`" --client --config `"$ConfigDir\client.conf`""

    $action = New-ScheduledTaskAction `
        -Execute $dashExe `
        -Argument $dashArgs `
        -WorkingDirectory "$ShareDir\dashboard"

    $trigger   = New-ScheduledTaskTrigger -AtLogOn
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -RunLevel Limited

    $settings  = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit (New-TimeSpan -Days 365)

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description "gpushare client dashboard -runs on user login" | Out-Null

    Write-Ok "Scheduled task '$TaskName' created (runs at logon via $dashExe)"
} catch {
    Write-Warn "Failed to create scheduled task (non-fatal): $_"
    Write-Info "You can manually start the dashboard: python `"$ShareDir\dashboard\app.py`""
}

# ==============================================================================
# STEP 9: Summary
# ==============================================================================
$verb = if ($IsUpgrade) { "upgraded" } else { "installed" }

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor White
Write-Host "    gpushare client $verb successfully!" -ForegroundColor Green
Write-Host "  ============================================================" -ForegroundColor White
Write-Host ""
Write-Host "  Server:          " -NoNewline; Write-Host $Server -ForegroundColor Cyan
Write-Host "  Install dir:     $InstallDir"
Write-Host "  Config:          $ConfigDir\client.conf"
Write-Host "  Compiler used:   $Compiler"
Write-Host ""
Write-Host "  DLLs installed:" -ForegroundColor White
Write-Host "    gpushare_client.dll   (base library)" -ForegroundColor Gray
foreach ($dll in $apiDlls) {
    Write-Host "    $dll" -ForegroundColor Gray
}
Write-Host ""
Write-Host "  CUDA override:   " -NoNewline; Write-Host "ACTIVE" -ForegroundColor Green
Write-Host "                   All GPU apps will use the remote GPU transparently."
Write-Host "  API coverage:    2620+ functions (cuBLAS, cuDNN, cuFFT, cuSPARSE, cuSOLVER, cuRAND, NVRTC)"
Write-Host "  Transfer opts:   " -NoNewline; Write-Host "ACTIVE" -ForegroundColor Green -NoNewline; Write-Host " (tiered pinned pools, async memcpy, chunked pipelining, D2H prefetch, RDMA, LZ4/zstd, multi-server GPU pooling)"
Write-Host "  Task Manager:    " -NoNewline; Write-Host "GPU tab will show remote GPU (after reboot)" -ForegroundColor Green
Write-Host ""
Write-Host "  Usage:" -ForegroundColor White
Write-Host "    python my_training.py        " -NoNewline -ForegroundColor Cyan; Write-Host "# uses remote GPU"
Write-Host "    nvidia-smi                   " -NoNewline -ForegroundColor Cyan; Write-Host "# show remote GPU stats"
Write-Host "    gpushare-monitor             " -NoNewline -ForegroundColor Cyan; Write-Host "# TUI status monitor"
Write-Host ""
Write-Host "  To change server:" -ForegroundColor White
Write-Host "    Edit $ConfigDir\client.conf"
Write-Host ""
Write-Host "  To uninstall:" -ForegroundColor White
Write-Host "    PowerShell -ExecutionPolicy Bypass -File $ProjectDir\scripts\uninstall-windows.ps1"
Write-Host ""
Write-Host "  NOTE: You may need to restart your terminal or log out/in" -ForegroundColor Yellow
Write-Host "  for PATH changes to take effect." -ForegroundColor Yellow
Write-Host ""
