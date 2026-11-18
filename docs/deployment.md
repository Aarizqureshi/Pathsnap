# Two-laptop deployment

| Laptop | Runs | Needs |
|---|---|---|
| 1: model + API | `start_api.bat` (FastAPI + uvicorn on port 8000) | repo, `pip install -r requirements.txt`, the evaluation trips, `models/final_model.joblib`, `configs/hmm.json` |
| 2: UI | `start_ui.bat` (static page on port 8080) | repo (only the `ui/` folder is used), Python for `http.server`, a browser with internet for the map tiles and Leaflet |

## Laptop 1

1. Keep the test sets side by side in one data folder, one sub-folder per trip:
   `Student folder\student_test\00000000\...` and `Student folder\evaluator_test\00000015\...`
2. From the repository root:
   ```
   start_api.bat "..\..\Student folder"
   ```
   Both `student_test` and `evaluator_test` are offered in the UI dropdown. `train` is never served.
   To offer only one set, pass it in quotes: `start_api.bat "..\..\Student folder" "student_test"`.
3. The window prints the laptop's IPv4 address. Allow Python through the Windows firewall if it asks (private network).
4. This window is the API log the evaluator must see. Every request is logged with the test set, trip id, timings, top-1 accuracy (as correct / total points), continuity, and the running overall accuracy.

## Laptop 2

1. `start_ui.bat`, then open `http://localhost:8080`.
2. Enter `http://<laptop-1 IPv4>:8000`, press **Connect**. The model panel shows the classifier, the decoder and their student-test accuracy.
3. Choose the test set in the dropdown, pick a trip, press **Predict route**. The map shows the GPS points and the predicted route. "This trip" shows server-side metrics for classifier-only vs classifier + HMM. "Overall accuracy" pools every trip run so far on that test set (correct / total points).

## What crosses the network

* UI to API: only `{"trip_id": "..."}`.
* API to UI: GPS points, predicted arc ids and route geometry, aggregate metrics, timings.
* The reference route never leaves laptop 1. Prediction runs on a copy of the trip with the route removed (`inference.strip_labels`). Scoring happens afterwards in `evaluation.py`.

## Checklist before the evaluation

- [ ] `python tests\test_hmm.py` passes on laptop 1
- [ ] `http://<ip>:8000/health` opens in laptop 2's browser
- [ ] All ten evaluation trips run once from the UI without errors
- [ ] Both laptops are on the same Wi-Fi or hotspot; the firewall allows port 8000
