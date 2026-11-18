# Two-laptop deployment

| Laptop | Runs | Needs |
|---|---|---|
| 1: model + API | `start_api.bat` (FastAPI + uvicorn on port 8000) | repo, `pip install -r requirements.txt`, the evaluation trips, `models/final_model.joblib`, `configs/hmm.json` |
| 2: UI | a browser only: open `http://<laptop-1 IP>:8000/` (the API serves the UI) | internet for the map tiles and Leaflet |

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

1. Open any browser at `http://<laptop-1 IPv4>:8000/`. The API on laptop 1 serves the UI itself, so laptop 2 needs no repo, no Python and no address to type. The page connects automatically. The model panel shows the classifier, the decoder and their student-test accuracy.
2. Choose the test set in the dropdown, pick a trip, press **Predict route**. The map shows the GPS points and the predicted route. "This trip" shows server-side metrics for classifier-only vs classifier + HMM. "Overall accuracy" pools every trip run so far on that test set (correct / total points).
3. Alternative (UI on its own web server): `start_ui.bat` on laptop 2, open `http://localhost:8080`, enter `http://<laptop-1 IPv4>:8000` and press **Connect**. Some browsers block this cross-address call or force it to https, so the first option is preferred.

## Troubleshooting

* Use laptop 1's Wi-Fi IPv4 address from `ipconfig`, not a `vEthernet (WSL)` / Hyper-V address.
* `http://<ip>:8000/health` must open on laptop 2. If not: same network on both (college Wi-Fi often blocks laptop-to-laptop traffic; use one phone hotspot), and allow port 8000 in the firewall (`netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000` as Administrator).
* "Invalid HTTP request received" in the API window means a browser sent https: type `http://`, and turn off Firefox HTTPS-Only Mode / Chrome "Always use secure connections".

## What crosses the network

* UI to API: only `{"trip_id": "..."}`.
* API to UI: GPS points, predicted arc ids and route geometry, aggregate metrics, timings.
* The reference route never leaves laptop 1. Prediction runs on a copy of the trip with the route removed (`inference.strip_labels`). Scoring happens afterwards in `evaluation.py`.

## Checklist before the evaluation

- [ ] `python tests\test_hmm.py` passes on laptop 1
- [ ] `http://<ip>:8000/` opens the UI in laptop 2's browser and connects
- [ ] All ten evaluation trips run once from the UI without errors
- [ ] Both laptops are on the same Wi-Fi or hotspot; the firewall allows port 8000
