"""Builds notebooks/01_data_audit.ipynb (T-02). Run: python scripts/make_audit_notebook.py"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
nb.cells = [
    md("# 01 - Data audit (PowerPulse Model 2)\n\nProfile schema, units, gaps, outliers and label coverage **before** modelling "
       "(PRISM-M task T-02). Set `CONFIG` and `RAW` below. The defaults use the SYNTHETIC demo data - "
       "switch to `configs/refit_house1.json` + your REFIT CSV for real work."),
    code("import sys; sys.path.insert(0, '..')\n"
         "import pandas as pd, numpy as np, matplotlib.pyplot as plt\n"
         "from src.config import load_config\nfrom src.ingest import load_raw\nfrom src.preprocess import clean_all\n"
         "from src.energy import detect_sessions, integrate_energy_kwh\n"
         "CONFIG = '../configs/synthetic_demo.json'\nRAW = '../data/raw/synthetic_house.csv'\n"
         "cfg = load_config(CONFIG)\nseries, meta = load_raw(RAW, cfg)\nmeta"),
    md("## 1. Quality profile (units: W, timestamps UTC)"),
    code("series, reports = clean_all(series, cfg)\n"
         "pd.DataFrame(reports).set_index('appliance')[['n_raw','n_negative','n_nan_or_non_numeric','n_duplicate_timestamps',"
         "'median_interval_s','n_gaps_over_max','longest_gap_h','coverage_pct','mean_w','max_w']].round(2)"),
    md("## 2. One day of power per appliance (visual sanity check of channel mapping - AC-03)"),
    code("day = series[next(iter(series))].index[0].normalize() + pd.Timedelta(days=30)\n"
         "fig, axes = plt.subplots(len(series), 1, figsize=(10, 1.6 * len(series)), sharex=True)\n"
         "for ax, (name, s) in zip(np.atleast_1d(axes), series.items()):\n"
         "    d = s[day: day + pd.Timedelta(days=1)]\n"
         "    ax.plot(d.index, d.values, lw=0.8); ax.set_ylabel(name, rotation=0, ha='right', fontsize=8)\n"
         "plt.xlabel('time (UTC)'); plt.tight_layout(); plt.show()"),
    md("## 3. Sessions and measured kWh per appliance"),
    code("sess = pd.concat([detect_sessions(s, cfg.appliances[n], cfg.dataset.max_gap_s) for n, s in series.items()])\n"
         "sess['baseline_kwh'] = sess['appliance'].map({n: a.rated_power_w for n, a in cfg.appliances.items()}) * sess['duration_h'] / 1000\n"
         "g = sess.groupby('appliance').agg(sessions=('duration_h','size'), median_duration_h=('duration_h','median'),\n"
         "    measured_kwh=('measured_energy_kwh','sum'), baseline_kwh=('baseline_kwh','sum'))\n"
         "g['baseline_over_measured'] = (g.baseline_kwh / g.measured_kwh).round(2)\ng.round(3)"),
    md("`baseline_over_measured` far from 1 means the rated power is a poor stand-in for real operating power (risk R-01)."),
    md("## 4. Threshold sensitivity (risk R-02)\n\nHow much do session count and energy change if the ON threshold is 0.5x / 1x / 2x?"),
    code("from dataclasses import replace\nname = list(series)[0]\nrows = []\n"
         "for f in (0.5, 1, 2):\n    c = replace(cfg.appliances[name], on_threshold_w=cfg.appliances[name].on_threshold_w * f)\n"
         "    x = detect_sessions(series[name], c, cfg.dataset.max_gap_s)\n"
         "    rows.append({'threshold_w': c.on_threshold_w, 'sessions': len(x), 'session_kwh': x.measured_energy_kwh.sum()})\n"
         "pd.DataFrame(rows)"),
    md("## 5. Total measured energy vs. energy inside detected sessions (standby energy outside sessions)"),
    code("tot = {n: integrate_energy_kwh(s, cfg.dataset.max_gap_s).energy_kwh for n, s in series.items()}\n"
         "inside = sess.groupby('appliance').measured_energy_kwh.sum()\n"
         "pd.DataFrame({'total_kwh': tot, 'in_sessions_kwh': inside}).assign(pct_in_sessions=lambda d: 100*d.in_sessions_kwh/d.total_kwh).round(2)"),
]
nbf.write(nb, "notebooks/01_data_audit.ipynb")
print("wrote notebooks/01_data_audit.ipynb")
