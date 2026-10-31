@echo off
setlocal

echo ================================================================
echo   Occlusion-Aware Face Recognition and Diagnostics Runner
echo ================================================================

set INPUT_IMG=%~1

if "%INPUT_IMG%"=="" (
    echo Usage: run_inference.bat ^<path_to_image^>
    echo Example: run_inference.bat dataset\sample\images\00000.png
    echo.
    set INPUT_IMG=dataset\sample\images\00000.png
)

if not exist checkpoints mkdir checkpoints

if not exist checkpoints\best_model.pth (
    echo Checkpoint not found locally.
    echo Downloading best_model.pth from GitHub Releases...
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://github.com/Pravin-kurosaki/segmentation-model/releases/download/v1.0.1/best_model.pth' -OutFile 'checkpoints\best_model.pth'"
)

python infer.py --image "%INPUT_IMG%" %2 %3 %4 %5

if %ERRORLEVEL% EQU 0 (
    echo.
    echo [SUCCESS] Diagnostic card saved to: results\inference_output.png
)
