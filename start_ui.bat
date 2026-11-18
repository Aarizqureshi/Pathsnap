@echo off
REM Laptop 2: serve the evaluator UI on http://localhost:8080 (run from the repository root).
echo Open http://localhost:8080 in a browser, enter http://<laptop-1 IP>:8000 and press Connect.
cd ui
python -m http.server 8080
