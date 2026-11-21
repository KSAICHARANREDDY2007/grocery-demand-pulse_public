from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

FORECAST_FILE = BASE_DIR / "artifacts" / "sku_forecasts.csv"
DASHBOARD_FILE = BASE_DIR / "inventory_dashboard.html"


# --------------------------------------------------
# LOAD FORECAST DATA
# --------------------------------------------------

if not FORECAST_FILE.exists():
    raise RuntimeError(
        "Missing artifacts/sku_forecasts.csv. "
        "Add the generated forecast CSV before deployment."
    )

forecasts = pd.read_csv(FORECAST_FILE)

required_columns = [
    "sku",
    "desc",
    "forecast_date",
    "forecast_next_day",
    "forecast_horizon",
    "sale_probability",
    "method",
    "avg_daily",
    "resid_std",
]

missing_columns = [
    col for col in required_columns
    if col not in forecasts.columns
]

if missing_columns:
    raise RuntimeError(
        f"Missing columns in forecast CSV: {missing_columns}"
    )

forecasts["sku"] = forecasts["sku"].astype(int)
forecasts["desc"] = forecasts["desc"].fillna("").astype(str)

forecasts = forecasts.set_index("desc", drop=False)


# --------------------------------------------------
# FASTAPI APPLICATION
# --------------------------------------------------

app = FastAPI(
    title="Grocery Demand Pulse API",
    description=(
        "Public API for next-day grocery demand forecasts "
        "and stockout-risk estimates."
    ),
    version="1.0.0",
)


# --------------------------------------------------
# CORS CONFIGURATION
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class PredictRequest(BaseModel):

    product: str = Field(
        ...,
        description="Product description or SKU ID"
    )


class StockoutRequest(BaseModel):

    product: str

    on_hand: Optional[float] = Field(
        default=None,
        ge=0
    )

    lead_time_days: int = Field(
        default=3,
        ge=1,
        le=60
    )

    service_z: float = Field(
        default=1.65,
        ge=0,
        le=4
    )


# --------------------------------------------------
# PRODUCT SEARCH HELPER
# --------------------------------------------------

def find_product(product: str):

    key = product.strip().casefold()

    matches = forecasts[
        forecasts["desc"].str.casefold() == key
    ]

    if matches.empty:

        if product.strip().isdigit():

            matches = forecasts[
                forecasts["sku"] == int(product.strip())
            ]

    if matches.empty:

        raise HTTPException(
            status_code=404,
            detail=(
                f"Product '{product}' not found. "
                "Use GET /search to find products."
            )
        )

    return matches.iloc[0]


# --------------------------------------------------
# FRONTEND HOMEPAGE
# --------------------------------------------------

@app.get("/")
def root():

    if not DASHBOARD_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="Frontend dashboard HTML file not found."
        )

    return FileResponse(
        path=DASHBOARD_FILE,
        media_type="text/html"
    )


# --------------------------------------------------
# DASHBOARD ROUTE
# --------------------------------------------------

@app.get("/dashboard")
def dashboard():

    if not DASHBOARD_FILE.exists():

        raise HTTPException(
            status_code=404,
            detail="Dashboard HTML file not found."
        )

    return FileResponse(
        path=DASHBOARD_FILE,
        media_type="text/html"
    )


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "ok",
        "project": "Grocery Demand Pulse",
        "products": int(len(forecasts)),
        "forecast_date": str(
            forecasts["forecast_date"].iloc[0]
        ),
        "source": "precomputed SKU forecasts",
    }


# --------------------------------------------------
# PRODUCTS
# --------------------------------------------------

@app.get("/products")
def products(
    limit: int = Query(
        default=20,
        ge=1,
        le=1000
    )
):

    data = forecasts.sort_values(
        "forecast_next_day",
        ascending=False
    ).head(limit)

    return [
        {
            "sku": int(row.sku),
            "product": row.desc,
            "forecast_next_day": round(
                float(row.forecast_next_day), 2
            ),
        }
        for row in data.itertuples()
    ]


# --------------------------------------------------
# PRODUCT SEARCH
# --------------------------------------------------

@app.get("/search")
def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(
        default=20,
        ge=1,
        le=200
    )
):

    hits = forecasts[
        forecasts["desc"].str.contains(
            q,
            case=False,
            na=False,
            regex=False
        )
    ].head(limit)

    return [
        {
            "sku": int(row.sku),
            "product": row.desc
        }
        for row in hits.itertuples()
    ]


# --------------------------------------------------
# DEMAND PREDICTION - GET
# --------------------------------------------------

@app.get("/predict")
def predict_get(
    product: str = Query(
        ...,
        description="Product description or SKU ID"
    )
):

    row = find_product(product)

    return {
        "sku": int(row.sku),
        "product": row.desc,
        "forecast_date": str(row.forecast_date),
        "forecast_next_day": round(
            float(row.forecast_next_day), 2
        ),
        "forecast_7_days": round(
            float(row.forecast_horizon), 2
        ),
        "sale_probability": round(
            float(row.sale_probability), 4
        ),
        "method": str(row.method),
    }


# --------------------------------------------------
# DEMAND PREDICTION - POST
# --------------------------------------------------

@app.post("/predict")
def predict_post(req: PredictRequest):

    row = find_product(req.product)

    return {
        "sku": int(row.sku),
        "product": row.desc,
        "forecast_date": str(row.forecast_date),
        "forecast_next_day": round(
            float(row.forecast_next_day), 2
        ),
        "forecast_7_days": round(
            float(row.forecast_horizon), 2
        ),
        "sale_probability": round(
            float(row.sale_probability), 4
        ),
        "method": str(row.method),
    }


# --------------------------------------------------
# STOCKOUT RISK
# --------------------------------------------------

@app.post("/stockout-risk")
def stockout_risk(req: StockoutRequest):

    row = find_product(req.product)

    on_hand = (
        float(req.on_hand)
        if req.on_hand is not None
        else round(float(row.avg_daily) * 5, 1)
    )

    lead_demand = (
        float(row.forecast_next_day)
        * req.lead_time_days
    )

    safety_stock = (
        req.service_z
        * float(row.resid_std)
        * np.sqrt(req.lead_time_days)
    )

    reorder_point = lead_demand + safety_stock

    if on_hand < lead_demand:
        risk = "HIGH"

    elif on_hand < reorder_point:
        risk = "MEDIUM"

    else:
        risk = "LOW"

    suggested_order = int(
        np.ceil(
            max(
                float(row.forecast_horizon)
                + safety_stock
                - on_hand,
                0
            )
        )
    )

    return {
        "sku": int(row.sku),
        "product": row.desc,
        "on_hand": round(on_hand, 2),
        "on_hand_is_simulated": req.on_hand is None,
        "lead_time_demand": round(lead_demand, 2),
        "safety_stock": round(safety_stock, 2),
        "reorder_point": round(reorder_point, 2),
        "risk": risk,
        "suggested_order": suggested_order,
    }


# --------------------------------------------------
# ALL FORECASTS
# --------------------------------------------------

@app.get("/forecasts")
def all_forecasts(
    limit: int = Query(
        default=100,
        ge=1,
        le=1000
    )
):

    data = forecasts.sort_values(
        "forecast_next_day",
        ascending=False
    ).head(limit)

    return data.replace(
        {np.nan: None}
    ).to_dict(orient="records")