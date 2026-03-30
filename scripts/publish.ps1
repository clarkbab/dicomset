$ErrorActionPreference = "Stop"

# Clean previous builds
if (Test-Path "dist") { Remove-Item -Recurse -Force "dist" }
if (Test-Path "build") { Remove-Item -Recurse -Force "build" }
if (Test-Path "*.egg-info" ) { Remove-Item -Recurse -Force "*.egg-info" }

# Build the package
Write-Host "Building package..." -ForegroundColor Cyan
python -m build

# Upload to PyPI
Write-Host "Uploading to PyPI..." -ForegroundColor Cyan
python -m twine upload dist/*

Write-Host "Done!" -ForegroundColor Green
