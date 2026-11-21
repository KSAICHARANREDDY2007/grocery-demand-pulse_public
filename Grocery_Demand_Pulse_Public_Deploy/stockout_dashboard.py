"""
Stockout-risk indicator + inventory dashboard.
Run AFTER the model notebook (needs artifacts/sku_forecasts.csv):   python stockout_dashboard.py
Creates: artifacts/stockout_risk.csv and inventory_dashboard.html
"""
import json
import numpy as np
import pandas as pd

# ---------- SIMULATED INVENTORY ASSUMPTIONS (the dataset has no stock-on-hand) ----------
PARAMS = dict(
    lead_time_days=3,      # supplier delivery time
    service_z=1.65,        # 1.28=90%, 1.65=95%, 2.33=99% service level
    cover_min=1.0,         # simulated stock on hand = avg daily sales x random days of cover
    cover_max=10.0,        #   drawn uniformly between cover_min and cover_max
    seed=42)

# ---------- STOCKOUT RULE ----------
# Lead-time demand = forecast_next_day x lead_time
# Safety stock     = z x forecast_error_std x sqrt(lead_time)
# Reorder point    = lead-time demand + safety stock
# HIGH   : on_hand <  lead-time demand      (will run out before a delivery arrives)
# MEDIUM : on_hand <  reorder point         (inside the safety-stock buffer)
# LOW    : on_hand >= reorder point
def stockout_risk(fc, p):
    d = fc.copy()
    rng = np.random.default_rng(p["seed"])
    d["on_hand"] = np.round(d["avg_daily"] * rng.uniform(p["cover_min"], p["cover_max"], len(d)), 1)
    L = p["lead_time_days"]
    d["lead_time_demand"] = d["forecast_next_day"] * L
    d["safety_stock"] = p["service_z"] * d["resid_std"] * np.sqrt(L)
    d["reorder_point"] = d["lead_time_demand"] + d["safety_stock"]
    d["days_of_cover"] = d["on_hand"] / d["forecast_next_day"].replace(0, np.nan)
    d["risk"] = np.where(d["on_hand"] < d["lead_time_demand"], "HIGH",
                 np.where(d["on_hand"] < d["reorder_point"], "MEDIUM", "LOW"))
    d["suggested_order"] = np.ceil(np.maximum(d["forecast_horizon"] + d["safety_stock"] - d["on_hand"], 0))
    return d

HTML = """<!DOCTYPE html><html><head><meta charset="utf-8"><title>Inventory Dashboard</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--bg:#f6f7f9;--card:#fff;--tx:#1d2733;--mu:#6b7785;--bd:#e3e7ec;--hi:#d62828;--me:#f4a261;--lo:#2a9d8f}
@media(prefers-color-scheme:dark){:root{--bg:#14181d;--card:#1e242b;--tx:#e8edf2;--mu:#98a3b0;--bd:#2d353e}}
body{margin:0;font-family:system-ui,Segoe UI,Arial,sans-serif;background:var(--bg);color:var(--tx);padding:20px}
h1{margin:0 0 4px;font-size:22px}.sub{color:var(--mu);font-size:13px;margin-bottom:16px}
.row{display:flex;gap:12px;flex-wrap:wrap;margin-bottom:14px}
.card{background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:14px 16px;flex:1;min-width:170px}
.k{font-size:28px;font-weight:700}.l{color:var(--mu);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
label{font-size:13px;color:var(--mu);display:block;margin-bottom:4px}input[type=range],select,input[type=text]{width:100%;box-sizing:border-box}
input[type=text],select{padding:6px;border-radius:6px;border:1px solid var(--bd);background:var(--card);color:var(--tx)}
.bar{display:flex;height:26px;border-radius:6px;overflow:hidden;margin-top:8px}.bar div{display:flex;align-items:center;justify-content:center;color:#fff;font-size:12px}
.tw{overflow-x:auto}table{width:100%;border-collapse:collapse;font-size:13px}th,td{padding:7px 8px;border-bottom:1px solid var(--bd);text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--mu);font-weight:600}
.b{padding:2px 8px;border-radius:10px;color:#fff;font-size:11px;font-weight:700}
</style></head><body>
<h1>Grocery Inventory Dashboard</h1><div class="sub">Forecast date __DATE__ &middot; on-hand stock is SIMULATED (dataset has no inventory records) &middot; change the settings to re-run the rule live</div>
<div class="row">
<div class="card"><label>Lead time: <b id="Lv"></b> days</label><input type="range" id="L" min="1" max="10" value="__L__"></div>
<div class="card"><label>Service level</label><select id="Z"><option value="1.28">90%</option><option value="1.65">95%</option><option value="2.33">99%</option></select></div>
<div class="card"><label>Show risk</label><select id="R"><option value="">All</option><option>HIGH</option><option>MEDIUM</option><option>LOW</option></select></div>
<div class="card"><label>Search product</label><input type="text" id="S" placeholder="e.g. MDH"></div></div>
<div class="row"><div class="card"><div class="l">High risk SKUs</div><div class="k" style="color:var(--hi)" id="kh"></div></div>
<div class="card"><div class="l">Medium risk SKUs</div><div class="k" style="color:var(--me)" id="km"></div></div>
<div class="card"><div class="l">Low risk SKUs</div><div class="k" style="color:var(--lo)" id="kl"></div></div>
<div class="card"><div class="l">Units to reorder (high+medium)</div><div class="k" id="ko"></div></div>
<div class="card" style="min-width:300px"><div class="l">Risk split</div><div class="bar" id="bar"></div></div></div>
<div class="card tw"><table><thead><tr><th>Product</th><th>Forecast/day</th><th>On hand (sim.)</th><th>Days cover</th><th>Reorder point</th><th>Order qty</th><th>Risk</th></tr></thead><tbody id="tb"></tbody></table>
<div class="sub" style="margin:8px 0 0">Showing the 40 most at-risk SKUs. Rule: HIGH = stock &lt; lead-time demand; MEDIUM = stock &lt; reorder point (lead-time demand + z&times;error&times;&radic;lead time).</div></div>
<script>
const D=__DATA__,H=__H__;const C={HIGH:'var(--hi)',MEDIUM:'var(--me)',LOW:'var(--lo)'};
const $=i=>document.getElementById(i);$('Z').value=__Z__;
function run(){const L=+$('L').value,z=+$('Z').value;$('Lv').textContent=L;
let rows=D.map(r=>{const[s,n,f,sd,oh]=r,ltd=f*L,ss=z*sd*Math.sqrt(L),rop=ltd+ss;
return{s,n,f,oh,rop,cov:f>0?oh/f:null,q:Math.ceil(Math.max(f*H+ss-oh,0)),k:oh<ltd?'HIGH':oh<rop?'MEDIUM':'LOW'}});
const c={HIGH:0,MEDIUM:0,LOW:0};rows.forEach(r=>c[r.k]++);
$('kh').textContent=c.HIGH;$('km').textContent=c.MEDIUM;$('kl').textContent=c.LOW;
$('ko').textContent=rows.filter(r=>r.k!='LOW').reduce((a,r)=>a+r.q,0).toLocaleString();
$('bar').innerHTML=['HIGH','MEDIUM','LOW'].map(k=>`<div style="width:${c[k]/rows.length*100}%;background:${C[k]}">${c[k]?c[k]:''}</div>`).join('');
const o={HIGH:0,MEDIUM:1,LOW:2},q=$('S').value.toLowerCase(),f=$('R').value;
rows=rows.filter(r=>(!f||r.k==f)&&r.n.toLowerCase().includes(q)).sort((a,b)=>o[a.k]-o[b.k]||b.f-a.f).slice(0,40);
$('tb').innerHTML=rows.map(r=>`<tr><td>${r.n}</td><td>${r.f.toFixed(2)}</td><td>${r.oh.toFixed(1)}</td><td>${r.cov==null?'-':r.cov.toFixed(1)}</td><td>${r.rop.toFixed(1)}</td><td>${r.q}</td><td><span class="b" style="background:${C[r.k]}">${r.k}</span></td></tr>`).join('')}
['L','Z','R','S'].forEach(i=>$(i).addEventListener('input',run));run();
</script></body></html>"""

if __name__ == "__main__":
    fc = pd.read_csv("artifacts/sku_forecasts.csv")
    res = stockout_risk(fc, PARAMS)
    res.to_csv("artifacts/stockout_risk.csv", index=False)
    print(res["risk"].value_counts().to_string())
    data = [[str(r.sku), r.desc.replace("<", ""), round(r.forecast_next_day, 3), round(r.resid_std, 3), r.on_hand]
            for r in res.itertuples()]
    html = (HTML.replace("__DATA__", json.dumps(data)).replace("__DATE__", str(fc["forecast_date"].iloc[0]))
            .replace("__L__", str(PARAMS["lead_time_days"])).replace("__Z__", f'"{PARAMS["service_z"]}"')
            .replace("__H__", "7"))
    open("inventory_dashboard.html", "w", encoding="utf-8").write(html)
    print("Saved artifacts/stockout_risk.csv and inventory_dashboard.html")
