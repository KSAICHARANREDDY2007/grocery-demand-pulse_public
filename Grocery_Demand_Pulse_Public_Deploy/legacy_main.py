"""Grocery Demand Pulse - FastAPI service.  Start:  python -m uvicorn main:app --reload"""
from typing import Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

bundle = joblib.load("demand_model.pkl")
fc = bundle["forecasts"].set_index("sku")          # one row per SKU: forecast, error std, history stats

app = FastAPI(title="Grocery Demand Pulse API",
              description="Next-day demand forecast and stockout risk per SKU")


class SkuRequest(BaseModel):
    product: str = "Aashirvad Atta 5kg"


class StockoutRequest(BaseModel):
    product: str = "Aashirvad Atta 5kg"
    on_hand: Optional[float] = None       # if empty, simulated as 5 days of average sales
    lead_time_days: int = 3
    service_z: float = 1.65               # 1.28=90%, 1.65=95%, 2.33=99%


def get_row(product: str):
    if product not in fc.index:
        raise HTTPException(status_code=404, detail=f"Product '{product}' not found (see GET /products)")
    return fc.loc[product]


@app.get("/")
def root():
    return {"message": "Grocery Demand Pulse API is running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "products": int(len(fc)), "model": bundle["best_regressor"],
            "forecast_date": str(fc["forecast_date"].iloc[0])}


@app.get("/products")
def products(limit: int = 10):
    top = fc.sort_values("forecast_next_day", ascending=False).head(limit)
    return [{"product": i, "forecast_next_day": round(float(r.forecast_next_day), 2)}
            for i, r in top.iterrows()]


@app.get("/search")
def search(q: str, limit: int = 10):
    """Find SKU ids by product name, e.g. /search?q=atta"""
    hit = fc[fc["desc"].str.contains(q, case=False, na=False)].head(limit)
    return [{"sku": int(i), "product": r.desc} for i, r in hit.iterrows()]


@app.post("/predict")
def predict(req: SkuRequest):
    r = get_row(req.product)
    return {"product": req.product, "forecast_date": str(r.forecast_date),
            "forecast_next_day": round(float(r.forecast_next_day), 2),
            "forecast_7_days": round(float(r.forecast_horizon), 2),
            "sale_probability": round(float(r.sale_probability), 2), "method": r.method}


@app.post("/stockout-risk")
def stockout_risk(req: StockoutRequest):
    r = get_row(req.product)
    on_hand = req.on_hand if req.on_hand is not None else round(float(r.avg_daily) * 5, 1)
    lead_demand = float(r.forecast_next_day) * req.lead_time_days
    safety = req.service_z * float(r.resid_std) * np.sqrt(req.lead_time_days)
    reorder_point = lead_demand + safety
    risk = "HIGH" if on_hand < lead_demand else "MEDIUM" if on_hand < reorder_point else "LOW"
    return {"product": req.product, "on_hand": on_hand,
            "on_hand_is_simulated": req.on_hand is None,
            "lead_time_demand": round(lead_demand, 2), "safety_stock": round(safety, 2),
            "reorder_point": round(reorder_point, 2), "risk": risk,
            "suggested_order": int(np.ceil(max(float(r.forecast_horizon) + safety - on_hand, 0)))}
