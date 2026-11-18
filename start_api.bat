@echo off
REM Laptop 1: start the PathSnap API (run from the repository root).
REM Usage:  start_api.bat "C:\path\to\data folder" [partitions]
REM   partitions = comma-separated test sets for the UI dropdown, in double quotes
REM                (default: student_test,evaluator_test; train is never served)

if "%~1"=="" (
  echo Usage: start_api.bat "C:\path\to\data folder" ["student_test,evaluator_test"]
  exit /b 1
)
set PATHSNAP_DATA=%~1
if "%~2"=="" (set PATHSNAP_PARTITIONS=student_test,evaluator_test) else (set PATHSNAP_PARTITIONS=%~2)
echo Data: %PATHSNAP_DATA%   test sets: %PATHSNAP_PARTITIONS%
echo Laptop 2: open  http://^<Wi-Fi IPv4 below^>:8000/  in any browser (the UI is served by this API).
echo Use the IPv4 address of the Wi-Fi adapter, not vEthernet/WSL:
ipconfig | findstr /i "IPv4"
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
