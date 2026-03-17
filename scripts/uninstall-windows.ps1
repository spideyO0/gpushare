#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Uninstall gpushare client from Windows.

.DESCRIPTION
    Removes all gpushare components: DLLs, config, registry entries (Task Manager GPU),
    scheduled tasks, PATH entries, Defender exclusions, and Python package.

.PARAMETER Purge
    Also remove configuration files (C:\ProgramData\gpushare).

.PARAMETER Yes
    Skip confirmation prompt.

.PARAMETER DryRun
    Show what would be removed without deleting anything.

.EXAMPLE
    .\uninstall-windows.ps1
    .\uninstall-windows.ps1 -Purge
    .\uninstall-windows.ps1 -Yes -Purge
    .\uninstall-windows.ps1 -DryRun
#>

param(
    [switch]$Purge,
    [switch]$Yes,
    [switch]$DryRun,
    [switch]$Help
)

# -- Helpers ------------------------------------------------------------------

function Write-Ok($msg)   { Write-Host "  [OK]    $msg" -ForegroundColor Green }
function Write-Info($msg)  { Write-Host "  [INFO]  $msg" -ForegroundColor Cyan }
function Write-Warn($msg)  { Write-Host "  [WARN]  $msg" -ForegroundColor Yellow }
function Write-Err($msg)   { Write-Host "  [ERROR] $msg" -ForegroundColor Red }
function Write-Step($msg)  { Write-Host "`n  --- $msg ---" -ForegroundColor White }
function Write-Action($action, $target) {
    $color = if ($DryRun) { "DarkGray" } else { "Red" }
    Write-Host "  $action  $target" -ForegroundColor $color
}

# -- Help ---------------------------------------------------------------------

if ($Help) {
    Get-Help $MyInvocation.MyCommand.Path -Detailed
    exit 0
}

# -- Admin check --------------------------------------------------------------

$isAdmin = ([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Err "This script must be run as Administrator."
    Write-Info "Right-click PowerShell -> Run as Administrator, then re-run."
    exit 1
}

# -- Paths --------------------------------------------------------------------

$InstallDir = "C:\Program Files\gpushare"
$ShareDir   = "C:\Program Files\gpushare\share"
$ConfigDir  = "C:\ProgramData\gpushare"
$TaskName   = "gpushare-dashboard"
$DisplayClassGuid = "{4d36e968-e325-11ce-bfc1-08002be10318}"
$ClassRegPath     = "HKLM:\SYSTEM\CurrentControlSet\Control\Class\$DisplayClassGuid"

# -- Banner -------------------------------------------------------------------

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor White
Write-Host "    gpushare Uninstaller (Windows)" -ForegroundColor White
Write-Host "  ============================================================" -ForegroundColor White
Write-Host ""

if ($DryRun) {
    Write-Warn "DRY RUN - nothing will be deleted"
    Write-Host ""
}

# -- Detect what's installed --------------------------------------------------

$found = $false

Write-Step "Scanning for gpushare components..."

# 1. Scheduled task
$hasTask = $false
try {
    $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($task) { $hasTask = $true; $found = $true }
} catch { }

if ($hasTask) {
    $state = $task.State
    Write-Action "REMOVE" "Scheduled task: $TaskName (state: $state)"
} else {
    Write-Info "Scheduled task '$TaskName' - not found"
}

# 2. Registry (virtual GPU for Task Manager)
$regEntries = @()
try {
    Get-ChildItem -Path $ClassRegPath -ErrorAction SilentlyContinue | ForEach-Object {
        $managed = (Get-ItemProperty -Path $_.PSPath -Name "gpushare_managed" -ErrorAction SilentlyContinue).gpushare_managed
        if ($managed -eq 1) {
            $drvDesc = (Get-ItemProperty -Path $_.PSPath -Name "DriverDesc" -ErrorAction SilentlyContinue).DriverDesc
            $regEntries += @{ Path = $_.PSPath; Desc = $drvDesc }
            $found = $true
        }
    }
} catch { }

if ($regEntries.Count -gt 0) {
    foreach ($entry in $regEntries) {
        Write-Action "REMOVE" "Registry GPU adapter: $($entry.Desc)"
    }
} else {
    Write-Info "Registry GPU adapter - not found"
}

# 3. Install directory
$hasInstallDir = Test-Path $InstallDir
if ($hasInstallDir) {
    $found = $true
    $files = Get-ChildItem -Path $InstallDir -Recurse -ErrorAction SilentlyContinue
    $dllCount = ($files | Where-Object { $_.Extension -eq ".dll" }).Count
    $totalSize = ($files | Measure-Object -Property Length -Sum -ErrorAction SilentlyContinue).Sum
    $sizeMB = if ($totalSize) { [math]::Round($totalSize / 1MB, 1) } else { 0 }
    Write-Action "REMOVE" ("$InstallDir - $dllCount DLLs, $sizeMB MB")
} else {
    Write-Info "$InstallDir - not found"
}

# 4. System PATH entry
$machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
$hasPathEntry = ($machinePath -split ";" | Where-Object { $_ -eq $InstallDir }).Count -gt 0
if ($hasPathEntry) {
    $found = $true
    Write-Action "REMOVE" "System PATH entry: $InstallDir"
} else {
    Write-Info "System PATH entry - not found"
}

# 5. Defender exclusion
$hasDefender = $false
try {
    $prefs = Get-MpPreference -ErrorAction SilentlyContinue
    if ($prefs.ExclusionPath -contains $InstallDir) {
        $hasDefender = $true
        $found = $true
        Write-Action "REMOVE" "Defender exclusion: $InstallDir"
    }
} catch { }
if (-not $hasDefender) {
    Write-Info "Defender exclusion - not found"
}

# 6. Python package
$hasPyPkg = $false
try {
    $pipOut = & python -m pip show gpushare 2>$null
    if ($LASTEXITCODE -eq 0 -and $pipOut) {
        $hasPyPkg = $true
        $found = $true
        Write-Action "REMOVE" "Python package: gpushare"
    }
} catch { }
if (-not $hasPyPkg) {
    Write-Info "Python package 'gpushare' - not found"
}

# 7. Startup shortcut (GPU tray widget)
$startupDir = [Environment]::GetFolderPath("Startup")
$trayShortcut = Join-Path $startupDir "gpushare GPU Monitor.lnk"
$hasTrayShortcut = Test-Path $trayShortcut
if ($hasTrayShortcut) {
    $found = $true
    Write-Action "REMOVE" "Startup shortcut: $trayShortcut"
} else {
    Write-Info "Startup shortcut - not found"
}

# 8. Running tray widget process
$hasTrayProcess = $false
$trayProcs = Get-Process -Name "pythonw","python" -ErrorAction SilentlyContinue |
    Where-Object { $_.CommandLine -and $_.CommandLine -match "gpu_tray_windows" }
if ($trayProcs) {
    $hasTrayProcess = $true
    $found = $true
    Write-Action "KILL" "GPU tray widget process (PID: $($trayProcs.Id -join ', '))"
} else {
    Write-Info "GPU tray widget process - not running"
}

# 8b. Processes using gpushare DLLs (the client library has an active recv thread)
$gpuProcs = @()
if ($hasInstallDir) {
    try {
        $gpuProcs = Get-Process -ErrorAction SilentlyContinue |
            Where-Object {
                try {
                    $_.Modules | Where-Object { $_.FileName -and $_.FileName -match "gpushare" }
                } catch { $false }
            } | Where-Object { $_ -ne $null }
    } catch { }
}
if ($gpuProcs.Count -gt 0) {
    $found = $true
    $procNames = ($gpuProcs | ForEach-Object { "$($_.Name) (PID $($_.Id))" }) -join ", "
    Write-Action "WARN" "Processes using gpushare DLLs: $procNames"
    Write-Info "These processes have active background threads from the gpushare DLL."
    Write-Info "They should be closed before uninstalling to avoid locked files."
} else {
    Write-Info "No processes using gpushare DLLs"
}

# 9. Config directory
$hasConfig = Test-Path $ConfigDir
if ($Purge -and $hasConfig) {
    Write-Action "REMOVE" "$ConfigDir [with -Purge]"
} elseif ($hasConfig) {
    Write-Host "  KEEP    $ConfigDir (use -Purge to remove)" -ForegroundColor Yellow
}

# Show DLL inventory if install dir exists
if ($hasInstallDir) {
    Write-Host ""
    Write-Host "  DLLs that will be removed:" -ForegroundColor Gray
    Get-ChildItem -Path $InstallDir -Filter "*.dll" -ErrorAction SilentlyContinue | ForEach-Object {
        Write-Host "    $($_.Name)" -ForegroundColor DarkGray
    }
}

# -- Nothing to do? -----------------------------------------------------------

if (-not $found) {
    Write-Host ""
    Write-Info "gpushare does not appear to be installed. Nothing to do."
    Write-Host ""
    exit 0
}

# -- Confirm ------------------------------------------------------------------

Write-Host ""

if ($DryRun) {
    Write-Warn "Dry run complete - nothing was removed."
    Write-Host ""
    exit 0
}

if (-not $Yes) {
    $answer = Read-Host "  Proceed with uninstall? [y/N]"
    if ($answer -ne "y" -and $answer -ne "Y") {
        Write-Info "Aborted."
        exit 0
    }
}

# -- Execute removal ---------------------------------------------------------

Write-Step "Removing gpushare..."

# 1. Stop and remove scheduled task
if ($hasTask) {
    try {
        Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction Stop
        Write-Ok "Removed scheduled task: $TaskName"
    } catch {
        Write-Warn "Could not remove scheduled task: $_"
    }
}

# 2. Remove registry GPU adapter
foreach ($entry in $regEntries) {
    try {
        Remove-Item -Path $entry.Path -Recurse -Force -ErrorAction Stop
        Write-Ok "Removed registry entry: $($entry.Desc)"
    } catch {
        Write-Warn "Could not remove registry entry: $_"
    }
}

# 3. Remove Defender exclusion
if ($hasDefender) {
    try {
        Remove-MpPreference -ExclusionPath $InstallDir -ErrorAction SilentlyContinue
        Write-Ok "Removed Defender exclusion"
    } catch {
        Write-Warn "Could not remove Defender exclusion: $_"
    }
}

# 4. Remove Python package
if ($hasPyPkg) {
    try {
        & python -m pip uninstall -y gpushare 2>$null | Out-Null
        Write-Ok "Removed Python package: gpushare"
    } catch {
        Write-Warn "Could not remove Python package: $_"
    }
}

# 5. Kill tray widget process
if ($hasTrayProcess) {
    try {
        $trayProcs | Stop-Process -Force -ErrorAction SilentlyContinue
        Write-Ok "Stopped GPU tray widget process"
    } catch {
        Write-Warn "Could not stop tray widget: $_"
    }
}

# 6. Remove startup shortcut
if ($hasTrayShortcut) {
    try {
        Remove-Item -Path $trayShortcut -Force -ErrorAction Stop
        Write-Ok "Removed startup shortcut"
    } catch {
        Write-Warn "Could not remove startup shortcut: $_"
    }
}

# 6b. Stop processes using gpushare DLLs (they have active recv threads)
if ($gpuProcs.Count -gt 0) {
    Write-Info "Requesting graceful exit from processes using gpushare DLLs..."
    foreach ($proc in $gpuProcs) {
        try {
            # Skip our own process
            if ($proc.Id -eq $PID) { continue }
            $proc.CloseMainWindow() | Out-Null
            if (-not $proc.WaitForExit(3000)) {
                $proc | Stop-Process -Force -ErrorAction SilentlyContinue
            }
            Write-Ok "Stopped $($proc.Name) (PID $($proc.Id))"
        } catch {
            Write-Warn "Could not stop $($proc.Name) (PID $($proc.Id)): $_"
        }
    }
    # Brief wait for DLL handles to release
    Start-Sleep -Milliseconds 500
}

# 7. Remove from system PATH (was step 5)
if ($hasPathEntry) {
    try {
        $newPath = ($machinePath -split ";" | Where-Object { $_ -ne $InstallDir -and $_ -ne "" }) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "Machine")
        Write-Ok "Removed from system PATH"
    } catch {
        Write-Warn "Could not update PATH: $_"
    }
}

# 8. Remove install directory (all DLLs, scripts, share)
if ($hasInstallDir) {
    try {
        Remove-Item -Path $InstallDir -Recurse -Force -ErrorAction Stop
        Write-Ok "Removed $InstallDir"
    } catch {
        Write-Warn "Could not fully remove $InstallDir - some files may be in use"
        Write-Info "The gpushare DLL uses background threads that may hold file locks."
        Write-Info "Close all GPU applications, wait a moment, then retry. Or reboot and retry."
    }
}

# 9. Remove config (only with -Purge)
if ($Purge -and $hasConfig) {
    try {
        Remove-Item -Path $ConfigDir -Recurse -Force -ErrorAction Stop
        Write-Ok "Removed $ConfigDir"
    } catch {
        Write-Warn "Could not remove config dir: $_"
    }
}

# -- Done ---------------------------------------------------------------------

Write-Host ""
Write-Host "  ============================================================" -ForegroundColor White
Write-Host "    gpushare uninstalled successfully." -ForegroundColor Green
Write-Host "  ============================================================" -ForegroundColor White
Write-Host ""

if (-not $Purge -and $hasConfig) {
    Write-Info "Configuration kept at $ConfigDir"
    Write-Info "Use -Purge to remove it too."
}

Write-Host ""
Write-Warn "Restart your terminal or log out/in for PATH changes to take effect."
if ($regEntries.Count -gt 0) {
    Write-Warn "Reboot to remove the GPU from Task Manager."
}
Write-Host ""
