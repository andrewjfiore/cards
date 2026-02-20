@echo off
REM ============================================================
REM  run_card_crop.bat
REM  Batch crop + straighten baseball card scans.
REM
REM  SETUP (one time):
REM    pip install opencv-python numpy pillow
REM
REM  Just drop your DSC*.JPG files in this folder and run.
REM ============================================================

:: ---- Settings ----------------------------------------------
SET INPUT_DIR=.
SET OUTPUT_DIR=output
SET BG=dark
:: Options: auto, dark, light
:: dark  = scanner bed is black (most flatbed scanners)
:: light = scanner bed or background is white

:: Optional: add white border around each card in pixels (0 = none)
SET PADDING=10

:: Uncomment to add --debug flag (saves contour overlay images)
:: SET DEBUG=--debug
SET DEBUG=
:: ------------------------------------------------------------

echo.
echo  ==========================================
echo   Baseball Card Batch Cropper
echo  ==========================================
echo   Input  : %INPUT_DIR%
echo   Output : %OUTPUT_DIR%
echo   BG mode: %BG%
echo.

python card_crop.py ^
    --input-dir  "%INPUT_DIR%"  ^
    --output-dir "%OUTPUT_DIR%" ^
    --bg         "%BG%"         ^
    --padding    "%PADDING%"    ^
    %DEBUG%

echo.
IF %ERRORLEVEL% EQU 0 (
    echo  Done! Results are in "%OUTPUT_DIR%"
    echo  If results look wrong, try editing BG=dark or BG=light in this .bat
) ELSE (
    echo  Error occurred - check output above.
)

echo.
pause
