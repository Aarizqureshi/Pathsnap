@echo off
REM Full PathSnap pipeline on Windows (run from the repository root).
REM Usage:  run_pipeline.bat "C:\path\to\Student folder"
REM The folder must contain train\ and student_test\ trip folders.

if "%~1"=="" (
  echo Usage: run_pipeline.bat "C:\path\to\data folder with train and student_test"
  exit /b 1
)
set PATHSNAP_DATA=%~1
echo Data: %PATHSNAP_DATA%

for %%S in (01_dataset_inventory 02_eda_plots 03_folium_maps 04_preprocess_and_label 05_build_features 06_feature_analysis 07_compare_models 08_bayesian_optimisation 09_hmm_decode 10_prediction_maps 11_summarise_results) do (
  echo.
  echo ===== %%S =====
  python scripts\%%S.py || (echo FAILED at %%S & exit /b 1)
)
python tests\test_hmm.py || exit /b 1
python tests\test_no_leakage.py || exit /b 1
echo.
echo All steps finished. Report tables: results\summary.md
