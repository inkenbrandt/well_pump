
Bayesian Pumping Reanalysis (Template)
=====================================

Created files:
- pumping_prepared.csv — cleaned table derived from your input.
- bayes_pumping_pymc.py — full PyMC model you can run locally.

How to run (locally):
1) Create/activate an environment with:  pip install pymc arviz numpy pandas
2) Put both files in the same directory.
3) Run:  python bayes_pumping_pymc.py
4) Outputs:
   - bayes_pumping_per_well.csv — per-well posterior stats.
   - bayes_pumping_totals.csv — total pumping posterior stats.

Model structure (high level):
- log(V_i) ~ N(alpha + beta*log(HP_i), sigma_v)
- Metered volume:  Site_acft_i ~ N(V_i, 5% RSE)
- Energy:          E_i ~ N(c_i * V_i, 15% RSE),  log(c_i) ~ N(mu_c, sigma_c)  (hierarchical)
- Hours:           H_i ~ N(V_i / flow_acft_per_hour_i, 20% RSE)

Tune the prior centers and relative SEs to match your instrumentation and site knowledge.
