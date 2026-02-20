@echo off
REM ============================================================
REM  run_card_crop.bat — Baseball Card Photo Cropper
REM
REM  Detects, crops, and straightens baseball cards from phone
REM  photos taken on a wood table.
REM
REM  SETUP (one time):
REM    pip install opencv-python numpy pillow
REM
REM  Drop your phone photos in INPUT_DIR and double-click this.
REM ============================================================

:: ---- SETTINGS (edit these) ----------------------------------

:: Folder containing your phone photos
SET INPUT_DIR=.

:: Folder where cropped cards will be saved
SET OUTPUT_DIR=output

:: White border added around each card, in pixels (0 = none)
SET PADDING=10

:: Debug mode: saves an image showing the detected contour in
:: green so you can diagnose failures.
:: Set to --debug to enable, or leave blank to disable.
SET DEBUG=
:: SET DEBUG=--debug

:: ---- END SETTINGS -------------------------------------------

echo.
echo  ==========================================
echo   Baseball Card Photo Cropper
echo  ==========================================
echo   Input  : %INPUT_DIR%
echo   Output : %OUTPUT_DIR%
echo   Padding: %PADDING%px
echo.

python card_crop.py ^
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
