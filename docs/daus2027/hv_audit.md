# HV audit (reviewer item R4, chunk C1)

Source: `scripts/rebuild_daus2027.py --only hv_audit` over `results_cobudget/` (read-only). Every number below is in one of the four `hv_*.csv` files beside this note. Seeds 0–29, 30 fronts per instance and arm.

## The five steps
1. **Fronts.** No export was needed: every run's final front is saved. The step loads 30 fronts per instance and arm and uses the distinct points (`aggregate._front_objs`), as the metrics did.
2. **Bounds** (`hv_bounds.csv`). Min–max over the reference front (the non-dominated union of NSGA-II + MOPSO): `lower_bound`/`upper_bound` per objective and instance, next to each arm's pooled best and worst. The reference front is identical with DMOPSO included, so the old two-arm and three-arm tables used one transformation.
3. **No clipping.** Checked that `normalize` returns exactly `(f − lo)/(hi − lo)`, with one `(lo, hi)` for all three arms. Every MOPSO point has normalized f₂ > 1.1: 2.79–7.86 (eil51), 1.53–5.82 (berlin52), 6.51–22.74 (eil76), 3.87–15.11 (rat99). NSGA-II also has values above 1.1 (`above_1_1`). The minimum is 0 on both axes.
4. **Points with no HV contribution** (`hv_points.csv`, leave-one-out per run).
   - At the old point (1.1, 1.1), exactly the points outside the box contribute nothing: NSGA-II 119/301, 127/745, 139/330 and 143/540 distinct points; MOPSO and DMOPSO all of theirs.
   - Old HV = 0: MOPSO in 30/30 runs on every instance; NSGA-II in 1 run (eil51) and 4 runs (eil76).
   - At the audited point: 0 points outside, 0 with no contribution and 0 runs with HV = 0, for both methods.
5. **Recomputed** with one common reference point (`hv_per_run.csv`), with sensitivity (`hv_sensitivity.csv`). The step first reproduces the paper's bounds and every per-run HV from `*_metrics.json` (to within 1e-12).

## Reference point chosen, and why
- Kept the min–max transformation and moved only the reference point: r = 1.1 × the per-objective worst normalized value, pooled over NSGA-II and MOPSO across all seeds. This gives eil51 (1.3647, 8.6429), berlin52 (1.3935, 6.3988), eil76 (1.5975, 25.0115) and rat99 (2.5826, 16.6253). HV is the fraction of the box [0, r₁] × [0, r₂], so it lies in [0, 1].
- Why: the point is common to both methods with the same transformation; every point of both methods lies strictly inside the box, so no front is truncated; and IGD, GD, SM and DM don't change.
- The factor 1.1 was fixed in prompt `09_C1` before any audited HV was computed, so the point was not picked after seeing which looks best.
- The old HV is **replaced**, not reported alongside: it was never defined as a deliberate region of interest. Only `hv_per_run.csv` is carried forward.

## Audited HV and sensitivity
- Factor 1.1, median (IQR), NSGA-II vs MOPSO: eil51 0.8820 (0.0519) vs 0.3202 (0.1887); berlin52 0.9018 (0.0338) vs 0.4271 (0.1721); eil76 0.9415 (0.0331) vs 0.4085 (0.2447); rat99 0.9458 (0.0298) vs 0.2920 (0.2242).
- Factors 1.05, 1.1, 1.2 and 1.5: A₁₂ = 1.000 on every instance at every factor. The magnitudes move (MOPSO's eil51 median goes from 0.2946 to 0.4751 between 1.05 and 1.5), but the order doesn't.
- **The NSGA-II vs MOPSO order held under every reference point tested**, including the old one and the three-method box. The stop rule was not triggered.

## What the old HV = 0 meant
- MOPSO's HV = 0 came from the reference point: every MOPSO point lies beyond (1.1, 1.1) on f₂. NSGA-II's 5 zeros are the same truncation. So a zero HV doesn't prove dominance on its own.
- Old values, here only, as box fractions (the paper's areas = fraction × 1.21). Median NSGA-II HV: 0.3364 (eil51), 0.5151 (berlin52), 0.2885 (eil76), 0.3204 (rat99). MOPSO: 0 on every instance. A₁₂: 0.983, 1.000, 0.933, 1.000.

## Reference-front shares (for R5)
- The reference front (9, 43, 15 and 33 points) is 100 % NSGA-II, with 0 % for MOPSO and DMOPSO (`ref_share_pct`). The R5 text must disclose this.
- This pooled result is not run-by-run dominance: keeping the word "dominated" would need C(A,B) (D3; the default is to reword).

## DMOPSO (Table VI, D7): audit note only
- Three-method box (r = 1.1 × worst over all three arms): it changes only on rat99, where r₁ = 3.6414 vs 2.5826 because DMOPSO's f₁ reaches 3.31. DMOPSO median HV: 0.5245, 0.5982, 0.5426 and 0.4272. NSGA-II vs DMOPSO A₁₂ = 1.000 on all four instances.
- Not quoted in the paper: adding DMOPSO would move the common box of the main comparison. On rat99, 1 DMOPSO point lies outside the two-method box.
