@echo off
REM ============================================================
REM  run_card_crop.bat — Baseball Card Photo Cropper
REM
REM  Detects, crops, and straightens baseball cards from phone
REM  photos taken on a wood table.
REM
REM  This script automatically creates a Python virtual environment
REM  and installs all required packages on first run.
REM
REM  Drop your phone photos in INPUT_DIR and double-click this.
REM ============================================================

:: ---- SETTINGS (edit these) ----------------------------------

:: Folder containing your phone photos
SET INPUT_DIR=.

:: Folder where cropped cards will be saved
SET OUTPUT_DIR=output

:: White border added around each card, in pixels (0 = none)
SET PADDING=0

:: Debug mode: saves an image showing the detected contour in
:: green so you can diagnose failures.
:: Set to --debug to enable, or leave blank to disable.
SET DEBUG=
:: SET DEBUG=--debug

:: ---- END SETTINGS -------------------------------------------

SET VENV_DIR=%~dp0.venv
SET REQUIREMENTS=%~dp0requirements.txt
SET PYTHON=%VENV_DIR%\Scripts\python.exe
SET PIP=%VENV_DIR%\Scripts\pip.exe

echo.
echo  ==========================================
echo   Baseball Card Photo Cropper
echo  ==========================================
echo   Input  : %INPUT_DIR%
echo   Output : %OUTPUT_DIR%
echo   Padding: %PADDING%px
echo.

:: ---- VENV SETUP --------------------------------------------

IF NOT EXIST "%PYTHON%" (
    echo [SETUP] Creating virtual environment in %VENV_DIR% ...
    python -m venv "%VENV_DIR%"
    IF %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment. Is Python installed?
        echo         Download from https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

:: Install / upgrade deps if requirements.txt is newer than our stamp file
SET STAMP=%VENV_DIR%\.deps_installed
IF NOT EXIST "%STAMP%" GOTO :install_deps
FOR /F %%A IN ('dir /b /od "%STAMP%" "%REQUIREMENTS%" 2^>nul') DO SET NEWEST=%%A
IF /I "%NEWEST%"=="requirements.txt" GOTO :install_deps
GOTO :deps_ok

:install_deps
echo [SETUP] Installing dependencies (this may take a few minutes on first run) ...
"%PIP%" install --upgrade pip >nul 2>&1
"%PIP%" install -r "%REQUIREMENTS%"
IF %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Dependency installation failed. Check the output above.
    pause
    exit /b 1
)
copy /y nul "%STAMP%" >nul 2>&1
echo [SETUP] Dependencies ready.
echo.

:deps_ok

:: ---- RUN ----------------------------------------------------

"%PYTHON%" "%~dp0card_crop.py" ^
    --input-dir  "%INPUT_DIR%"  ^
    --output-dir "%OUTPUT_DIR%" ^
    --padding    "%PADDING%"    ^
    %DEBUG%

echo.
IF %ERRORLEVEL% EQU 0 (
    echo  Done! Results are in "%OUTPUT_DIR%"
) ELSE (
    echo  Error occurred - check output above.
)

echo.
pause
