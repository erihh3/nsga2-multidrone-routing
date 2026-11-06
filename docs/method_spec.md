# Method specification (DAUS 2027 revision)

Describes what the code in `src/uav/` does, as run for the paper's data (`results_cobudget/`). Written for reviewer items R2 (decoder), R1 (what is shared, initialization, fleet moves), R3 (configuration of each method) and, in a later section, the budget.

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

## Shared and not shared

Reviewer item R1. A common evaluator doesn't equalize representation, operators, initialization or the ease of changing the fleet. This section states exactly what the two compared methods, NSGA-II and MOPSO, share and what they don't. Code: `src/uav/algorithms/{nsga2,mopso}.py`, `src/uav/problem/`.

### Shared (the same code in both methods)

- **Instance and distance matrix:** `uav.problem.instance.load_instance`, `inst.dist`.
- **Phenotype and objectives:** (visit order, counts) → `_partition_to_routes` → K routes → `uav.problem.fitness.evaluate` → (f₁, f₂). See "Decoder" above.
- **Evaluation counting:** one `CountingEvaluator` class (`uav.problem.fitness`) wraps every fitness call in both methods. The budget itself is A3's section.
- **Seeds and seeding:** seeds 0–29. `set_all_seeds(seed)` seeds Python's `random` and NumPy's legacy RNG at the top of each `run()`.
- **Population and swarm size:** 100.
- **Downstream:** metrics, statistics and plots read only the saved fronts and routes (`uav.evaluation.*`, `uav.viz.*`) and never branch on the method.

### Not shared

| Aspect | NSGA-II | MOPSO |
|---|---|---|
| Genotype | Permutation of `0..N-1` plus integer `counts` of length K | Random keys in `[0,1]^(N+K)` |
| Genotype → (order, counts) | None: the genotype *is* (order, counts) | **Keys → (permutation, counts), which exists only in MOPSO**: argsort of the N order keys, and `_counts_from_keys` on the first K−1 fleet keys |
| Initialization | Uniform permutation; two cuts uniform on `{0..N}` (below) | U(0,1) keys, zero velocity; cuts `round(U·N)` (below) |
| Variation | OX on the permutation; one-point crossover on `counts` + `_repair_counts`; inversion; `shift` / `consolidate` on `counts` | Velocity update (inertia, cognitive, social) with a velocity clamp; reflective bound; turbulence |
| Selection and elitism | `selNSGA2` (rank + crowding) on parents + offspring after `_dedup`; `selTournamentDCD` for mating | pbest rule; leader drawn by adaptive-grid roulette from an external archive; archive truncation in the most crowded cell |
| Fleet moves | Explicit count moves: `shift`, `consolidate`, counts crossover | None: counts move only through the fleet keys |
| Final set | First non-dominated front of the last population | The external archive (≤ 100). Detail in A3 |

### Initialization (the first RNG use after `set_all_seeds`)

- **NSGA-II** (`nsga2._init_individual`, called by `toolbox.population(n=100)`):
  - The permutation is `random.shuffle` of `0..N-1`: a uniform permutation.
  - The counts come from `_random_composition`: two cuts drawn independently with `random.randint(0, N)`, i.e. uniform on the N+1 values `{0..N}`, then sorted; the counts are the gaps.
  - Each ordered pair of cuts has probability 1/(N+1)². So a composition with two distinct cuts is twice as likely as one with equal cuts, and the docstring's "uniform random composition" is not exact. The code is not changed.
- **MOPSO** (`mopso.py:216-218`):
  - Positions are `np.random.uniform(0, 1, size=(100, N+K))`, and velocities start at zero.
  - Argsort of i.i.d. U(0,1) order keys gives a uniform permutation.
  - Each cut is `round(U·N)`. The interior values `1..N-1` each have probability 1/N. The endpoints 0 and N each have 1/(2N), because only half of their rounding interval lies inside [0, 1].
- Both methods start from uniform permutations. Their initial fleet distributions differ (see "Initial fleet frequencies" below).

### How each method changes its fleet

- **NSGA-II changes the counts directly.**
  - *Counts crossover:* with probability `pcx` = 0.9 per mating pair, one-point crossover on the two `counts` vectors (point in `1..K-1`). `_repair_counts` then adds or removes single units at random drones until the sum is N. A child can gain or lose an empty route.
  - *Count mutation* (effective probability 0.15 per individual per generation; see "Configuration of each method"):
    - `shift` moves one POI from a random non-empty donor to another drone. It empties a route whose count is 1 and re-activates an empty one.
    - `consolidate` moves the donor's whole count onto another drone (`c_dst += c_src; c_src = 0`), so it empties one route in a single step.
  - **`shift` and `consolidate` are count moves that MOPSO lacks.**
- **MOPSO has no count operator.** Its counts change only when a velocity update or a turbulence reset moves a fleet key's `round(key·N)` across a rounding boundary.
  - A key moves at most `vmax` = 0.1 per iteration, i.e. at most 0.1·N cut positions (5 at N = 50).
  - Emptying a route needs both cut keys to round to the same value, or one cut key within 0.5/N of 0 or 1. Cut 0 empties drone 1, and cut N empties drone 3.
  - The reflective bound clips a key that leaves [0, 1] to exactly 0.0 or 1.0, which gives cut 0 or N and so empties the first or last drone. It also flips that velocity component.
  - Turbulence redraws keys from U(0,1). The K-th fleet key is inert (see "Decoder").

### Initial fleet frequencies

Source: `scripts/rebuild_daus2027.py --only initial_fleet` → `docs/daus2027/initial_fleet_frequencies.csv`.

- **Method.** For each instance, method and seed 0–29, the step replays the run's first RNG use without running the loop:
  - NSGA-II: `set_all_seeds`, then `toolbox.population(n=100)`.
  - MOPSO: `set_all_seeds`, then `np.random.uniform(0, 1, (100, N+K))`.
- **Active routes** are counted with `len(r) > 2`.
- **Columns.** "obs" is the mean over the 30 seeds of each seed's share, in %. "exact" is the formula below, in %.
- **Replay check.** In all 240 runs, the objectives of the replayed population reproduce the run's saved gen-0 best, mean and worst (`history[0]`, relative tolerance 1e-9). These are therefore the populations the runs actually started from.

| Instance | N | Method | Individuals | gen-0 replay | 1 route (obs / exact) | 2 routes (obs / exact) | 3 routes (obs / exact) |
|---|---|---|---|---|---|---|---|
| eil51-k3 | 50 | NSGA-II | 3000 | 30/30 | 0.133 / 0.154 | 9.000 / 9.419 | 90.867 / 90.427 |
| eil51-k3 | 50 | MOPSO | 3000 | 30/30 | 0.000 / 0.040 | 5.733 / 5.880 | 94.267 / 94.080 |
| berlin52-k3 | 51 | NSGA-II | 3000 | 30/30 | 0.167 / 0.148 | 8.667 / 9.246 | 91.167 / 90.607 |
| berlin52-k3 | 51 | MOPSO | 3000 | 30/30 | 0.067 / 0.038 | 5.033 / 5.767 | 94.900 / 94.195 |
| eil76-k3 | 75 | NSGA-II | 3000 | 30/30 | 0.100 / 0.069 | 6.500 / 6.406 | 93.400 / 93.525 |
| eil76-k3 | 75 | MOPSO | 3000 | 30/30 | 0.000 / 0.018 | 3.733 / 3.947 | 96.267 / 96.036 |
| rat99-k3 | 98 | NSGA-II | 3000 | 30/30 | 0.067 / 0.041 | 5.133 / 4.948 | 94.800 / 95.011 |
| rat99-k3 | 98 | MOPSO | 3000 | 30/30 | 0.000 / 0.010 | 3.167 / 3.030 | 96.833 / 96.960 |

Exact probabilities for K = 3 (two cuts):
- NSGA-II: P₁ = 4/(N+1)², P₂ = 5(N−1)/(N+1)², P₃ = (N−1)(N−2)/(N+1)².
- MOPSO: P₁ = 1/N², P₂ = 3(N−1)/N², P₃ = (N−1)(N−2)/N².

**Agreement.** The observed shares agree with the formulas within sampling noise. The largest |z| over the 24 cells is 1.72 (binomial SE, 3000 individuals per cell).

**The 1-route case.** It is too rare to estimate from 3000 samples: 1.2 to 4.6 expected per instance for NSGA-II, 0.3 to 1.2 for MOPSO. Use the formulas for it.

**Reading.**
- At initialization MOPSO has fewer 1- and 2-route individuals than NSGA-II on every instance.
- P₁ is about 4 times smaller in MOPSO.
- P₂ is (3/5)(1+1/N)² times NSGA-II's, about 0.6.
- In both methods, at least 90 % of the initial individuals use all three routes.

### Verdict on "matching density"

`peer_review/paper.tex` lines 363–364: "The cut-point construction makes idle drones ($c_k=0$) first-class outcomes, matching NSGA-II's fleet-sampling density".

**Verdict: FALSE. Mark "matching NSGA-II's fleet-sampling density" for deletion (E3b).**
- **Per cut:** NSGA-II puts 1/(N+1) on each of `0..N`. MOPSO puts 1/N on each interior value and 1/(2N) on 0 and on N. Those endpoints are exactly where a cut empties the first or last drone.
- **Consequence:** P₁ is about 4 times lower in MOPSO and P₂ about 3/5 of NSGA-II's (table above).
- **The first half of the sentence is true:** idle drones are reachable outcomes (see "Decoder").
- **Suggested replacement for E3b:** "The cut-point construction makes idle drones ($c_k=0$) reachable outcomes, although at initialization it yields fewer one- and two-route solutions than NSGA-II's composition sampler, because cut values 0 and N carry half weight."
- **The same claim in the code:** the `_counts_from_keys` docstring (`src/uav/problem/decode.py:64-65`, "the same fleet-sampling density as the two-part chromosome") is just as inexact. It's a code comment and is left unchanged, since this revision makes no `src` edits.

### Limitation (D2)

D2 default: the matched-initial-phenotype control is not run. Text for E3b and E3a:

> The two methods share the decoder, the objectives, the evaluation counter and the budget, but not the representation, the variation operators, the initialization or the way the fleet changes: NSGA-II's initial population holds more one- and two-route solutions, and only NSGA-II has explicit count moves (shift, consolidate). No control with matched initial phenotypes was run. The differences reported here are therefore differences between the tested implementations as configured, and are not attributed to the GA or PSO family alone.

### R1 sentences to change

**Search.** The search ran on a whitespace-normalized `peer_review/paper.tex`, so sentences that wrap across lines are still found.
- Terms: `attribut`, `licens`, `exactly one thing`, `co-equal`, `strictly`, `optimizer and its genotype`, `optimizer alone`, plus `structural`, `family`, `inherent`, `only difference`, `genotype alone`, `solely`.
- Left out, because they are not claims: the `%%` header comment (lines 3–4); `family` at 201–203 (metric family), 475–479 (family-wise error) and 600–605 (Holm family).
- Added by reading the context: lines 738–740 and 768–770, which make the same attribution without any of the search terms.

**Owners.** E3a covers the title, abstract, introduction, related work and conclusion. E3b covers the formulation and methods. E45 covers the results and discussion.

| Lines | Sentence (quoted, shortened) | Why it overclaims (R1) | Action | Chunk |
|---|---|---|---|---|
| 40–41 | Title: "A Co-Equal Comparison of NSGA-II and MOPSO …" | "Co-equal" implies everything but the optimizer is equalized | Replace with the D6 title | E3a |
| 65–67 | "… MOPSO collapses to the full fleet and never reaches the energy-optimal low-fleet region given the co-equal comparison." | Same claim, in the abstract | Drop "given the co-equal comparison" | E3a |
| 67 | "This weakness appears structural: with a random-key encoding, MOPSO cannot reliably produce the near-optimal single tours …" | Attributes the gap to the encoding; initialization and count moves also differ | Restrict to the tested implementation | E3a |
| 97–102 | "The study address that gap with a co-equal comparison. NSGA-II and MOPSO are made to differ in exactly one thing, that is the optimizer loop and its genotype, while sharing …" | False: they also differ in initialization, operators, fleet moves and selection/archive | Replace with the shared / not-shared statement | E3a |
| 119–122 | "Compare NSGA-II and MOPSO co-equally: … so any observed difference is attributable to the optimizer and its genotype alone" | Attribution to the optimizer alone | List what is shared; no attribution | E3a |
| 127–132 | "Identifies and explains a structural limitation of random-key MOPSO … to preserve a pure-GA-vs-pure-PSO comparison" | Family-level attribution ("six times the budget" falls under D4) | Restrict to the tested MOPSO; apply D4 | E3a |
| 250–252 | "(b) the co-equal NSGA-II-vs-MOPSO comparison" | Same claim | "a comparison under a shared decoder, objectives and budget" | E3b |
| 281–282 | "This single rule is what licenses attributing any performance difference to the optimizer and its genotype." | A shared evaluator doesn't license that attribution (the reviewer's point) | Delete; add the D2 limitation | E3b |
| 363–364 | "matching NSGA-II's fleet-sampling density" | False (verdict above) | Delete; use the replacement above | E3b |
| 396–400 | "… would require a matching local search in NSGA-II to stay co-equal" | Implies the pair is co-equal now | "to keep the comparison controlled" | E3b |
| 523 | "Co-equality requires an equal evaluation budget, …" | Same term | "A fair comparison requires …" | E3b |
| 683–686 | "kept canonical random-key MOPSO … as the co-equal baseline" | Same term | Drop "co-equal" (DMOPSO text per D7) | E45 |
| 738–740 | "The order encoding is therefore not the binding constraint: … pointing instead to a search-dynamics limitation" | Attributes to PSO search dynamics; initialization and missing count moves untested | Restrict; mention the untested differences | E45 |
| 746–749 | "Under a strictly co-equal protocol, NSGA-II significantly outperformed MOPSO …" | Same claim | "Under the shared decoder, objectives and budget, …" | E45 |
| 768–770 | "The limitation is thus in the search dynamics since the swarm cannot keep an initially dominated low-fleet solution … in any single representation choice." | Attributes to the PSO family; the initial fleet share and count moves also differ | Add the D2 caveat; restrict to the tested implementation | E45 |
| 781–784 | "… preserving the co-equal comparison would then require an equivalent local search in NSGA-II …" | Same term | "keeping the comparison controlled" | E45 |
| 793–802 | "The study compares NSGA-II and MOPSO co-equally, … traced … to a search-dynamics limitation rather than the encoding alone" | Same claims, in the conclusion | Reword; restrict; add the D2 limitation | E3a |
| 802–806 | "… a matching local search added to both arms to retain co-equality" | Same term | "to keep the comparison controlled" | E3a |

## Configuration of each method

Reviewer item R3. Values come from `hyperparams` in the run JSONs of `results_cobudget/`, which are identical across instances and seeds. Meanings come from the code. `config.py`'s defaults (`gens=500`, `iters=500`) are not the paper's.

### NSGA-II (`src/uav/algorithms/nsga2.py`)

| Parameter | Value | Effective meaning | Code |
|---|---|---|---|
| `pop` | 100 | Population size; also the number of offspring per generation | `run`, `selTournamentDCD(pop, hp.pop)` |
| `gens` | 3000 | Generations | `run` |
| `pcx` | 0.9 | Per mating pair: OX on the permutation **and** one-point crossover on `counts` (+ `_repair_counts`), applied together | `run`, `_mate` |
| `pmut` | 0.3 | Per offspring: mutation fires. Inversion of a random sub-tour always runs inside it | `run`, `_mutate` |
| `pmut_counts` | 0.5 | Conditional on `pmut`: count mutation fires. **Effective count-mutation rate 0.3 × 0.5 = 0.15 per offspring per generation** | `_mutate` |
| shift / consolidate | 50 / 50 | Within a count mutation, so each runs at an effective 0.075 per offspring | `_mut_counts` |
| Selection | none | `selNSGA2` on parents + offspring, after `_dedup` (which only reorders, so one representative per objective vector comes first); binary `selTournamentDCD` for mating | `run`, `_dedup` |
| Evaluation | none | Only offspring changed by crossover or mutation are evaluated; unchanged clones keep their fitness, so `n_evals` varies by seed | `run` |

### MOPSO (`src/uav/algorithms/mopso.py`)

| Parameter | Value | Effective meaning | Code |
|---|---|---|---|
| `swarm` | 100 | Particles | `run` |
| `iters` | 2790 | Iterations; the whole swarm is evaluated each iteration plus once at start: 100 × 2791 = 279,100 evaluations | `run` |
| `w_inertia`, `c1`, `c2` | 0.4, 1.5, 2.0 | v ← w·v + c1·r1·(pbest − x) + c2·r2·(leader − x), with r1, r2 ~ U(0,1) drawn per key | `run`, lines 240–246 |
| `vmax_frac` | 0.1 | Each velocity component is clamped to ±0.1 (key range 1); initial velocity is 0 | `run`, line 247 |
| Bound | none | Reflective: positions are clipped to [0, 1] and the velocity components that left are flipped | lines 249–252 |
| `mut_rate`, `mut_floor` | 0.3, 0.05 | **Per-particle probability** of turbulence at iteration t: max(0.05, 0.3·(1 − t/T)). It reaches the floor at t = 5T/6 ≈ 2325 | `_turbulence_fraction`, line 253 |
| Turbulence size | none | Each hit resets a **fixed** `max(1, int(0.1·(N+K)))` keys, drawn without replacement, to U(0,1): 5, 5, 7 and 10 keys on eil51, berlin52, eil76 and rat99 | `_turbulence` |
| pbest rule | none | Replaced if the new position dominates it; if neither dominates, replaced with probability 0.5 | lines 264–268 |
| `archive_size` | 100 | External archive of non-dominated, objective-deduplicated members. Above 100, a random member of the most crowded grid cell is dropped until it fits | `_rebuild_archive`, `_truncate` |
| `grid_divisions` | 30 | Per objective, over the archive's own min/max. The leader is drawn by roulette over cells (weight 10/\|cell\|), then uniformly within the cell; one draw per particle per iteration | `_leader_sampler`, `_draw_leader` |

**Order in the code, per particle:** leader → velocity update → velocity clamp → move → reflect into [0, 1] → turbulence (lines 239–254).

**Text to fix (for E3b):**
- Table I, "Mutation prob. (Part 2) 0.5": the effective rate is 0.15 per individual (0.5 is conditional on `pmut` = 0.3).
- Line 393–396: the paper says turbulence resets "a fraction of keys" and that "this fraction decays". In the code, the decaying quantity is the per-particle probability, and each hit resets a fixed `max(1, int(0.1·(N+K)))` keys.
- Algorithm 2 (line 414): the paper says "apply decaying turbulence; reflect into [0,1]". The code does the opposite: reflect first, then turbulence.

**DMOPSO (diagnostic arm only):** the same swarm hyperparameters as MOPSO (identical `hyperparams` in its JSONs), with a permutation plus K cut keys decoded by `decode_discrete`. It is not part of the main comparison.
