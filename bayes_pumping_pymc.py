
# Bayesian model for annual pumping using PyMC (template)
# Save as bayes_pumping_pymc.py and run in an environment with PyMC installed.
# Expects "pumping_prepared.csv" in the same directory.

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az

DATA_CSV = "pumping_prepared.csv"
df = pd.read_csv(DATA_CSV)

# Safe log helper
def safelog(x):
    return np.log(np.clip(x, 1e-12, None))

hp = df["horsepower"].astype(float).to_numpy()
vol_obs = df["site_acft"].astype(float).to_numpy()
E_obs = df["total_kwh"].astype(float).to_numpy()
H_obs = df["sitehours"].astype(float).to_numpy()
flow_gpm = df["flow_gpm_for_hours"].astype(float).to_numpy()
flow_af_per_hr = df["flow_acft_per_hour"].astype(float).to_numpy()

# Masks
has_vol = np.isfinite(vol_obs)
has_E = np.isfinite(E_obs)
has_H = np.isfinite(H_obs) & np.isfinite(flow_af_per_hr) & (flow_af_per_hr > 0)

# Fill missing/invalid HP with median to keep regression usable; uncertainty is handled by sigma_v
hp_filled = hp.copy()
valid_hp = np.isfinite(hp_filled) & (hp_filled > 0)
median_hp = np.nanmedian(hp_filled[valid_hp]) if valid_hp.any() else 50.0
hp_filled[~valid_hp] = median_hp

with pm.Model() as model:
    # Prior: log V ~ N(alpha + beta*log(HP), sigma_v)
    alpha = pm.Normal("alpha", mu=0.0, sigma=5.0)
    beta  = pm.Normal("beta",  mu=1.0, sigma=2.0)
    sigma_v = pm.HalfNormal("sigma_v", sigma=1.5)

    logV_prior_mean = alpha + beta * pm.math.log(hp_filled + 1e-12)
    logV = pm.Normal("logV", mu=logV_prior_mean, sigma=sigma_v, shape=len(df))
    V = pm.Deterministic("V", pm.math.exp(logV))

    # kWh per ac-ft factor (hierarchical lognormal): c_i ~ LogNormal(mu_c, sigma_c)
    mu_c = pm.Normal("mu_c", mu=np.log(350.0), sigma=1.0)   # center ~350 kWh/ac-ft
    sigma_c = pm.HalfNormal("sigma_c", sigma=0.8)
    logc = pm.Normal("logc", mu=mu_c, sigma=sigma_c, shape=len(df))
    c = pm.Deterministic("c", pm.math.exp(logc))

    # Metered volume: tight 5% RSE around V
    if has_vol.any():
        pm.Normal("vol_like",
                  mu=V[has_vol],
                  sigma=pm.math.maximum(0.05 * V[has_vol], 1e-4),
                  observed=vol_obs[has_vol])

    # Energy: E ~ N(c*V, 15% RSE)
    if has_E.any():
        muE = V[has_E] * c[has_E]
        pm.Normal("energy_like", mu=muE,
                  sigma=pm.math.maximum(0.15 * muE, 1.0),
                  observed=E_obs[has_E])

    # Hours: H ~ N(V / flow_af_per_hr, 20% RSE)
    if has_H.any():
        muH = V[has_H] / flow_af_per_hr[has_H]
        pm.Normal("hours_like", mu=muH,
                  sigma=pm.math.maximum(0.20 * muH, 0.5),
                  observed=H_obs[has_H])

    # Sample
    trace = pm.sample(1200, tune=1200, target_accept=0.9, chains=4)

    # Summaries
    # Stack chains and draws for V
    v = trace.posterior["V"]
    v_stack = v.stack(sample=("chain","draw")).values   # (n_wells, n_samples)
    total_samples = v_stack.sum(axis=0)

    print("Total pumping (ac-ft) posterior:")
    for q in [5, 25, 50, 75, 95]:
        print(f"  {q}%: ", float(np.percentile(total_samples, q)))

    # Save per-well summary
    v_mean = np.mean(v_stack, axis=1)
    v_p05 = np.percentile(v_stack, 5, axis=1)
    v_p95 = np.percentile(v_stack, 95, axis=1)
    out = df[["well_id"]].copy()
    out["V_mean_acft"] = v_mean
    out["V_p05_acft"] = v_p05
    out["V_p95_acft"] = v_p95
    out.to_csv("bayes_pumping_per_well.csv", index=False)

    totals = pd.DataFrame({
        "total_p05_acft":  [float(np.percentile(total_samples, 5))],
        "total_p25_acft":  [float(np.percentile(total_samples, 25))],
        "total_p50_acft":  [float(np.percentile(total_samples, 50))],
        "total_p75_acft":  [float(np.percentile(total_samples, 75))],
        "total_p95_acft":  [float(np.percentile(total_samples, 95))],
        "total_mean_acft": [float(np.mean(total_samples))],
    })
    totals.to_csv("bayes_pumping_totals.csv", index=False)
