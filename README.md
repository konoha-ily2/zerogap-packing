# zerogap-packing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23630728.svg)](https://doi.org/10.5281/zenodo.23630728)

Exact contact analysis for packings of convex polyhedra in convex polyhedral containers:
closed forms of the container scale, exact verification of touching configurations, and first-order local-optimality certificates.

Author: **Hirotaka Nakano** ([konoha-ily2](https://github.com/konoha-ily2))
First published: 2026-10 (see the commit history for exact dates)

The records analysed here come from the [Polyhedra in Polyhedra](https://github.com/alejandrozu/platonic-packing) catalogue (MIT License).
Their coordinates and the exact solid definitions (`src/solids_exact.json`) are downloaded at run time and are not redistributed here.
Code comments are in Japanese.

## What it does

| Script | Purpose |
|---|---|
| `algebraic3d.py` | Detects the contacts of a packing (vertex–wall, vertex–face, edge–edge), writes them as polynomial equations in the scale *S*, the centres and the unit quaternions, and solves them to ~117 digits by mixed-precision Newton refinement. The kernel of the Jacobian tells whether *S* is fixed by the contacts (rigidity). `mpmath.findpoly` then finds an integer polynomial for *S*, and `sympy` turns degree ≤ 4 cases into radicals. |
| `closed_batch.py` | Runs `algebraic3d.py` over many catalogue records in parallel and compares the results with the catalogue's `closed_form_conjecture` column. |
| `certify_touch.py` | Exact check of a configuration whose pieces may **touch** (zero gap), in Q(√2) arithmetic: containment, and a separating plane for every pair. Interval-based certifiers that require a positive gap cannot accept these. |
| `round_touch.py` | Rounds a catalogue configuration to exact numbers at the exact closed-form scale and passes it to `certify_touch.py`. |
| `numfield.py`, `touchk.py` | The same, extended to multi-quadratic fields Q(√2, √3, √5, …) (e.g. dodecahedra and icosahedra). |
| `force_cert.py` | First-order local-optimality certificate: non-negative contact forces with zero net force and torque on every piece (Farkas' lemma), solved and checked in exact arithmetic. |

## Results (snapshot)

- `data/closed_forms/` — one JSON per record: contact counts, equation rank, kernel dimension, *S* to ~117 digits, minimal polynomial and closed form when found.
  Only records whose scale is fixed by their contacts are included (1,044 records in 25 piece/container pairs, as of v1.2.0; this now includes improved configurations that the catalogue has accepted).
- `RESULTS.tsv` — summary table of the above.
  - Counts below are as of v1.1.1 (2026-10-25) and have not been recomputed for v1.2.0. **455 closed forms**, of which **257 are not yet in the catalogue's `closed_form_conjecture` column** and 198 agree with it (mostly to 116–117 digits; configurations at a singular minimum of a contact-preserving family give 17–28 digits).
- `data/exact_touch/` — exact touching configurations and their checks (v1.1.1 count: 266 records EXACT-VALID at the exact closed-form scale; `summary.tsv` in Q(√2), `summary_k.tsv` in multi-quadratic fields).
- `data/exact_touch/force_summary.tsv` — force certificates (v1.1.1 counts): 22 strict local minima, 6 local minima up to sliding pieces, 115 first-order certificates where the second-order test is inconclusive.

## Usage

Python ≥ 3.10 with `numpy`, `mpmath` and `sympy`.

```
python algebraic3d.py cubincub_n37          # downloads the record and solids into data/ on first use
python closed_batch.py unknown --jobs 8     # all records without a catalogue closed form
python algebraic3d.py cubincub_n29          # round_touch.py uses the record and closed form fetched/written here
python round_touch.py cubincub_n29          # exact touching check at the closed-form scale
python force_cert.py data/exact_touch/cubincub_n29.json
```

## Citation

If you use these results or methods, please cite this repository (see `CITATION.cff`):

> Hirotaka Nakano. *zerogap-packing: exact contact analysis, closed forms and local-optimality certificates for polyhedral packings.* Zenodo. https://doi.org/10.5281/zenodo.23630728

This DOI always resolves to the latest version; v1.0.0 is https://doi.org/10.5281/zenodo.23630729.

## License

MIT (see `LICENSE`).

The exact configurations in `data/exact_touch/` are roundings of catalogue records, which are distributed under the MIT License,
Copyright (c) 2026 Yohei Nakajima (soft-to-rigid-packing: method, simulator, tightening and checker design),
Copyright (c) 2026 Alejandro Zarzuelo (generalization to Platonic solids in Platonic containers).
The `source` field of each file names the record it was derived from.
