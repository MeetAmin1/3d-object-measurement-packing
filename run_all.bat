@echo off
setlocal
cd /d "%~dp0"

echo.
echo ===== Part 1: Oriented Bounding Boxes =====
python src\part1_obb_measurement.py --input-dir data --output-dir outputs --save-video outputs\part1_obb_demo.mp4
if errorlevel 1 exit /b 1

echo.
echo ===== Part 2: Gravity-Aware Packing =====
python src\part2_gravity_packing.py --items "data\Item List.json" --output-dir outputs --save-video outputs\part2_packing_demo.mp4
if errorlevel 1 exit /b 1

echo.
echo ===== Independent Validation =====
python src\validate_submission.py
if errorlevel 1 exit /b 1

echo.
echo ===== Unit Tests =====
python -m unittest discover -s tests -v
if errorlevel 1 exit /b 1

echo.
echo ===== ALL CHECKS PASSED =====
exit /b 0
