# Method specification (DAUS 2027 revision)

Describes what the code in `src/uav/` does, as run for the paper's data (`results_cobudget/`). Written for reviewer items R2 (decoder) and, in later sections, the shared core and budget.

## Decoder

Reviewer item R2. Code: `src/uav/problem/decode.py`; genotypes in `src/uav/algorithms/{nsga2,mopso,dmopso}.py`. N = number of POIs (depot excluded), K = 3 drones.

### Genotype of each method

| Method | Genotype | Decoder |
|---|---|---|
| NSGA-II | Permutation of the 0-based positions `0..N-1`, plus an integer vector `counts` of length K with sum N and each entry ≥ 0. Crossover and mutation on `counts` end with `_repair_counts`, which restores that invariant. No keys. | `decode_two_part` |
| MOPSO | Random-key vector in `[0,1]^(N+K)`. | `decode_random_key` |
| DMOPSO (diagnostic arm) | Permutation of the 0-based positions `0..N-1`, plus K continuous cut keys `kappa` in `[0,1]`. | `decode_discrete` |

**The keys → (permutation, counts) step exists only in MOPSO.** DMOPSO uses only the keys → counts half (`_counts_from_keys`). NSGA-II has no keys: its order and counts are the genotype itself.

### Shared phenotype builder `_partition_to_routes(order, counts, depot)`

All three decoders end here, so all methods share one phenotype space.

1. Raises `ValueError` if `sum(counts) != N` or any count is negative.
2. Cuts `order` into K contiguous segments of lengths `counts[0], …, counts[K-1]`, in that order.
3. Bookends each segment with the depot: `[depot, p_1, …, p_c, depot]`. A zero count gives the route `[depot, depot]`. That drone stays at the depot, so the route has distance 0 and adds nothing to makespan or energy.

The output is always K routes. Every POI appears exactly once, and every route leaves and returns to the depot.

### POI id mapping

`load_instance` stores node 1 of the `.tsp` file (the depot) at index 0. POI ids are therefore `1..N`, built as `poi_ids = [node for node in range(N+1) if node != depot]`. NSGA-II and DMOPSO permutations hold 0-based positions, and position `g` maps to `poi_ids[g]` (= `g+1`). MOPSO's argsort index `j` maps the same way.

### MOPSO: `decode_random_key(keys, N, K, depot)`

- The length must be exactly N+K; otherwise it raises `ValueError`.
- **Order keys** (`keys[0..N-1]`): `np.argsort(keys[:N], kind="stable")` gives the visit order over POI ids.
- **Fleet keys** (`keys[N..N+K-1]`) go to `_counts_from_keys`, below.

### Counts from keys: `_counts_from_keys(fleet_keys, N, K)` (MOPSO and DMOPSO)

The steps, in order:

1. **Role of each key.** Only the first K−1 fleet keys are read (`[:K-1]`). Each one is one cut position on the integer line `[0, N]`.
2. **The extra key.** The K-th fleet key (key number N+K overall) is **inert**: K−1 cuts already determine K counts. It is kept only so the particle position and velocity arrays, and the length check in `decode_random_key`, stay at N+K. PSO velocity updates and MOPSO's turbulence still move it, with no effect on routes or objectives. (A turbulence draw that lands on it changes nothing.) This is why the dimension is N+K although the text describes K−1 cuts.
3. **Clipping.** The keys are clipped to `[0, 1]` (`np.clip`). The PSO position update already clips to `[0, 1]` and turbulence draws from U(0,1), so this only guards the decoder's own contract.
4. **Rounding.** `cut = int(round(key * N))` uses Python's `round` on a float, which rounds **half to even**. Worked examples at N = 5: key 0.5 → 2.5 → cut **2** (rounds down to even); key 0.7 → 3.5 → cut **4** (rounds up to even). Keys within `[0, 1]` give cuts within `[0, N]`.
5. **Cut ordering.** The cuts are sorted ascending, so the order of the keys doesn't matter: keys `[0.7, 0.5]` and `[0.5, 0.7]` both give cuts `[2, 4]`.
6. **Counts.** The counts are the gaps of `[0, *cuts, N]`: `c_k = bound[k+1] − bound[k]`. Because the cuts are sorted and lie in `[0, N]`, every count is ≥ 0 and the counts sum to N. Example: cuts `[2, 4]` at N = 5 give counts `[2, 2, 1]`.
7. **Repeated cuts and zero counts.**
   - Two equal cuts give a zero count for the drone between them.
   - A cut at 0 empties the first drone; a cut at N empties the last.
   - Both cuts at the same endpoint give the **single-route case**: `[0,0]` → `[0,0,N]`, `[N,N]` → `[N,0,0]`.
   - Every composition of N into K non-negative parts is reachable, with cut fractions = cumulative counts / N.

   A zero count becomes the route `[depot, depot]` (see above).

### Ties

The order keys are sorted with `np.argsort(..., kind="stable")`, so **tied keys keep POI-index order** (lower POI id first). Ties are not a measure-zero event: clipping in the PSO update piles keys up at exactly 0.0 or 1.0, so this rule is used in practice. The rule is deterministic. Cut ties are the "repeated cuts" case above.

### Out of scope here

The fact that an empty route *can* occur doesn't show that it occurs with equal probability in each method. That comparison belongs to A2 (`initial_fleet_frequencies.csv`).

### Suggested text for the manuscript (for E3b; the manuscript is not edited in A1)

The text gives dimension N+K but describes K−1 cut keys. One sentence closes the gap, for example: "Only the first K−1 of the K fleet keys are read, and the last is unused (it keeps the vector length at N+K). Cuts use round-half-to-even, sorted cuts may coincide or fall at 0 or N, giving idle drones, and tied order keys keep POI-index order."

### Minimal check (reviewer's request)

- **Small examples** (`tests/test_decode.py`, N = 5, K = 3):
  - Every one of the 21 compositions of 5 into 3 non-negative parts, decoded by both `decode_two_part` and `decode_random_key` (cut fractions = cumulative counts / N). This covers the single-route cases (5,0,0), (0,5,0) and (0,0,5) and every empty-route pattern.
  - Each output is checked for: 3 routes, depot at both ends, no depot inside a route, each POI exactly once, and decoded counts equal to the composition (so they are ≥ 0 and sum to N).
  - The random-key output equals the two-part output, and it is identical whether the inert key is 0.0 or 1.0.
  - One tie case (all order keys equal → POIs in index order), and one rounding case (keys `[0.7, 0.5]` → cuts `[2, 4]` → counts `[2, 2, 1]`).
  - Result: 55 tests pass (10 earlier, 45 new).
- **Saved solutions** (`scripts/rebuild_daus2027.py --only check_solutions` → `docs/daus2027/solution_check.csv`): all 360 runs (4 instances × 3 methods × seeds 0–29), **13,042 solutions, 0 failures**. Each solution is checked for coverage, depot bookends and K routes, and both objectives are recomputed with `uav.problem.fitness.evaluate` (relative tolerance 1e-9).

Decoder verdict: MATCH
