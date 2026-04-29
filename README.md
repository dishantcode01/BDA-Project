# Earthquake Prediction Project

This project predicts:

- Earthquake **magnitude** (regression)
- Earthquake **risk level** (classification: Low / Medium / High)

It includes:

- Model training and evaluation for **Linear**, **Polynomial**, **Logarithmic**, and **Power** regression
- Risk classification with a decision tree
- Flask API for predictions and metrics
- Frontend dashboard
- MySQL schema for storing predictions

## Project Structure

```
earthquake_project/
├── dataset/
├── model/
│   ├── artifacts/
│   └── train.py
├── backend/
│   └── app.py
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── database/
│   └── schema.sql
├── notebooks/
└── report/
    └── charts/
```

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Put your dataset in `dataset/` as either:

- `USG_earthquake_data.csv` (preferred)
- `earthquake.csv`

If multiple CSV files exist, the training script will auto-pick one.

3. Train models and generate metrics/charts:

```bash
python model/train.py
```

4. (Optional) Create MySQL database/table:

```bash
mysql -u root -p < database/schema.sql
```

5. Start backend (single app mode, serves frontend too):

```bash
python backend/app.py
```

6. Open app:

Open `http://127.0.0.1:5000` in browser.

## Deployment Notes

- The Flask app now serves frontend pages and API from the same domain.
- For production, run:

```bash
gunicorn backend.app:app
```

- Platform configs:
  - `Procfile` included: `web: gunicorn backend.app:app`
  - `requirements.txt` includes `gunicorn`

## API Endpoints

- `GET /` - Frontend app home page
- `GET /health` - API health
- `GET /metrics` - Returns RMSE, MAE, R2, and classification accuracy
- `POST /predict` - Predict magnitude and risk
- `GET /generated-charts/<filename>` - Serve generated chart images

Example `/predict` body:

```json
{
  "lat": 19.07,
  "lon": 72.88,
  "depth": 10.4,
  "model": "linear"
}
```

`model` accepts: `linear`, `polynomial`, `logarithmic`, `power`.

## Notes

- Risk labels are derived from magnitude:
  - `< 4` => Low
  - `< 6` => Medium
  - `>= 6` => High
- Chart images are saved in `report/charts/`.
