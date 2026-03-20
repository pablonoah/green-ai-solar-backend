# Green AI Solar -- Backend

FastAPI backend serving a GradientBoosting regression model to predict solar panel efficiency based on environmental and panel parameters.

**Part of the [Green AI Solstice](https://github.com/noahlips/Green_AI_Solstice) project -- ESILV**
**Frontend:** [solar-app](https://github.com/noahlips/solar-app)

## Tech Stack

- **Framework:** FastAPI
- **ML:** scikit-learn (GradientBoostingRegressor)
- **Data:** Pandas, NumPy
- **Serialization:** Joblib
- **Language:** Python 3.11

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API status |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict solar panel efficiency |
| `GET` | `/feature-importance` | Get feature importance ranking |
| `GET` | `/model-info` | Model metadata (R2=0.814) |

## Model

- **Algorithm:** GradientBoostingRegressor (100 estimators, depth 5)
- **R2 Score:** 0.814
- **Input features:** temperature, irradiance, humidity, panel age, soiling ratio, voltage, current, cloud coverage, wind speed, pressure, and more
- **Top predictors:** irradiance (67%), soiling ratio (23%), panel age (8%)

## Getting Started

```bash
git clone https://github.com/noahlips/green-ai-solar-backend.git
cd green-ai-solar-backend
pip install -r requirements.txt

# Train the model
python train_model.py

# Start the API
uvicorn main:app --reload --port 8000
```

API docs available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Deployment

Deployed on [Render](https://render.com) via `render.yaml`.
