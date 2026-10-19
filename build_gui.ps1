param(
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'

Push-Location $PSScriptRoot
try {
    $pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
    if (-not (Test-Path $pythonExe)) {
        $pythonExe = (Get-Command python).Source
    }

    $pyInstallerArgs = @(
        '--noconfirm',
        '--clean',
        '--windowed',
        '--onedir',
        '--name', 'AvalancheSegmentationToolkit',
        '--paths', '.',
        '--paths', 'segmentation',
        '--add-data', 'gui\segmentation_gui_config.yaml;gui',
        '--add-data', 'segmentation\smoke_seg\full50\weights;segmentation\smoke_seg\full50\weights',
        '--add-data', 'segmentation\smoke_seg\full150\weights;segmentation\smoke_seg\full150\weights',
        '--add-data', 'segmentation\yolov5n-seg.pt;segmentation',
        'gui\segmentation_gui.py'
    )

    if ($Clean) {
        Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
    }

    & $pythonExe -m PyInstaller @pyInstallerArgs
}
finally {
    Pop-Location
}
