# Plate-with-a-hole stress dataset — v1 (steel, linear elastic, ANSYS)

380 finite-element simulations of a 2D steel plate with one circular hole, made in ANSYS Mechanical
APDL 2023 R1 for the IUE thesis project *"Predicting cracks in composite structures with machine
learning"* (stage 1: isotropic steel, von Mises stress). Every case contains the full nodal stress
field, the mesh, and the stresses around the hole edge.

**Goal for ML:** predict the stress (especially the peak at the hole edge, Kt) from 4 design
parameters, without running FE.

---

## 1. Start in 2 minutes

```bash
pip install pandas numpy scikit-learn matplotlib pyarrow
python examples/quickstart.py      # loads all three data levels and trains a first model
python examples/plot_case.py 106   # draws mesh + von Mises field of case 106
```

---

## 2. What is in the folder

| Path | What it is | Size |
|---|---|---|
| `data/cases.csv` | **One row per case**: inputs, derived features, targets (Kt, peak angle), train/test split. Start here. | 84 kB |
| `data/hole_profiles.npz` | Stresses on the hole edge, 360 points per case (1° grid), all cases in one array | 5 MB |
| `data/hole_profiles_long.csv` | Same as above in long CSV form (for Excel / MATLAB) | 7 MB |
| `data/fields/nodes_all.parquet` | **Full stress field**: every corner node of every case (3.39 M rows) | 114 MB |
| `data/fields/elements_all.parquet` | Mesh connectivity of every case (3.27 M rows) | 55 MB |
| `simulation/` | ANSYS macros and the parameter tables used to generate the data (regenerate / extend it) | — |
| `validation/` | Evidence the FE model is correct: mesh convergence, analytical checks, quality checks | — |
| `figures/` | Example meshes and stress fields | — |
| `examples/` | `quickstart.py`, `plot_case.py` | — |

Parquet files open with `pandas.read_parquet` (Python) or `parquetread` (MATLAB R2019a+).
Read only one case with `pd.read_parquet(path, filters=[("case", "==", 106)])`.

---

## 3. The physical model (same for all cases)

```
 y ▲            top edge:   σy = λ·σx
 100 ┌──────────────────────────────────────────┐
     │                                          │
σx ◄─┤                 ◯  hole                  ├─► σx      plate 300 × 100 × 2 mm
     │          centre (150+ex, 50+ey)          │
   0 └──────────────────────────────────────────┘
     0  A           bottom edge: σy = λ·σx    B 300 ─► x
```

| Item | Value |
|---|---|
| Units | **mm, N, MPa** |
| Coordinates | origin at bottom-left corner A; x along the 300 mm length, y along the 100 mm width |
| Plate | L = 300 mm, W = 100 mm, thickness t = 2 mm |
| Material | Steel, linear elastic isotropic: E = 200 GPa, ν = 0.3 (no plasticity, no yield) |
| Analysis | 2D static, plane stress, PLANE183 (8-node quadratic) |
| Loads | Uniform edge traction: σx = **100 MPa** tension on left and right edges; σy = λ·100 MPa on top and bottom edges (λ < 0 → compression) |
| Supports | Rigid-body only: corner A (0,0) UX = UY = 0, corner B (300,0) UY = 0. Loads are self-balanced, reactions ≈ 1e-8 N |
| Mesh | 1° elements on the hole edge (360 around), 3 mm far field, top/bottom edges refined to ligament/6 |

**Why every stress is "per 100 MPa":** the model is linear and loaded by tractions, so all stresses
scale exactly with load, and E and ν do not affect them. Divide any stress by 100 to get the
dimensionless stress-concentration value (columns ending in `_n`, and `kt`); multiply by your real
nominal stress to get MPa. Valid only while the material stays elastic — several cases exceed the
yield stress of real steel at 100 MPa nominal load.

---

## 4. The four varied parameters and the splits

| Parameter | Column | Meaning | Range |
|---|---|---|---|
| D | `D` [mm], `dw` = D/W | hole diameter | D/W 0.10 – 0.60 (`test_extrap`: 0.60 – 0.65) |
| ex | `ex` [mm], `exw` = ex/W | hole x-offset from plate centre (+ = right) | −0.5 W … +0.5 W |
| ey | `ey` [mm], `eyw` = ey/W | hole y-offset from plate centre (+ = up) | up to ±0.2 W, limited so each ligament ≥ 10 mm |
| λ | `lam` | load ratio σy/σx | −1 (pure shear) … 0 (uniaxial) … +1 (equibiaxial) |

Sampled with Latin hypercube sampling (separate seeds per split, no duplicates between splits):

| `split` | Cases | Case IDs | Use it for |
|---|---|---|---|
| `train` | 300 | 1 – 300 | training **and** all tuning (use cross-validation inside these 300) |
| `test_interp` | 50 | 1001 – 1050 | final test, same parameter ranges |
| `test_extrap` | 30 | 2001 – 2030 | final test outside the training range (larger holes) |

**Rules so results are comparable and honest**
1. Never train or tune on `test_interp` / `test_extrap`. Look at them once, at the end.
2. When you use the field data (one row per node), split **by case**, never by node — nodes of the
   same plate are near-duplicates and a node-level split gives fake, inflated scores.
3. Report errors in physical terms (% error of Kt, MPa), not only R². Most of the plate sits near
   the nominal stress, so field-level R² looks excellent even when the peak is wrong.

---

## 5. Column dictionary

### `data/cases.csv` — one row per case

| Column | Unit | Description |
|---|---|---|
| `case` | – | case ID (folder/file key everywhere) |
| `split` | – | `train`, `test_interp`, `test_extrap` |
| `D`, `ex`, `ey` | mm | hole diameter, x-offset, y-offset (see §4) |
| `lam` | – | load ratio σy/σx |
| `dw`, `exw`, `eyw` | – | D/W, ex/W, ey/W |
| `lig_top_w`, `lig_bot_w` | – | ligament (hole edge → top / bottom plate edge) divided by W |
| `lig_min_w` | – | smaller of the two ligaments / W |
| `lig_ratio` | – | smaller ligament / larger ligament (1 = hole centred in y) |
| `net_w` | – | net-section width (W − D)/W |
| `dist_end_w` | – | distance hole edge → nearest loaded x-end / W |
| `kt_kirsch` | – | analytical infinite-plate peak for this λ: max(\|3 − λ\|, \|3λ − 1\|) (a feature, not a target) |
| **`kt`** | – | **main target**: max von Mises on the hole edge / σx |
| `theta_pk` | deg | angle of that peak on the hole (convention in §6) |
| `kt_global` | – | max von Mises anywhere in the plate / σx (≠ `kt` in 5 cases, see §8) |
| `peak_on_hole` | bool | True if the plate maximum is on the hole edge |
| `second_peak_ratio` | – | highest hole-edge value > 45° away from the peak, divided by the peak (≥ 0.99 = two near-equal peaks) |
| `n_nodes`, `n_elem` | – | mesh size of this case |
| `mesh_err_pct` | % | ANSYS energy-norm error estimate (all ≤ 0.12 %) |

### `data/hole_profiles.npz` — hole edge, 1° grid

| Key | Shape | Description |
|---|---|---|
| `case` | (380,) | case IDs, same order as rows of the arrays (and of `cases.csv`) |
| `theta_deg` | (360,) | −180 … 179 |
| `seqv` | (380, 360) | von Mises / σx |
| `stt` | (380, 360) | hoop (tangential) stress σθθ / σx — the only non-zero stress on a free hole edge |
| `sx`, `sy`, `sxy` | (380, 360) | global stress components / σx |

`hole_profiles_long.csv` has the same data with columns `case, theta_deg, seqv_n, stt_n, sx_n, sy_n, sxy_n`.

### `data/fields/nodes_all.parquet` — full field

| Column | Unit | Description |
|---|---|---|
| `case` | – | case ID |
| `node_id` | – | ANSYS node number (**unique only within a case**; every case has its own mesh) |
| `x`, `y` | mm | node position (origin bottom-left corner) |
| `sx`, `sy`, `sxy` | MPa | stress components, global x-y axes |
| `s1`, `s2` | MPa | in-plane principal stresses |
| `seqv` | MPa | von Mises stress |

Corner nodes only (midside nodes of the quadratic elements are not exported). Stresses are ANSYS
averaged nodal values. Applied σx is 100 MPa in every case.

### `data/fields/elements_all.parquet` — mesh

`case, elem_id, n1 … n8`: n1–n4 are the corner nodes (counter-clockwise), n5–n8 midside nodes
(not in `nodes_all`). A few elements are triangles: then `n3 == n4`. To draw or interpolate the
mesh use n1–n4 only (see `examples/plot_case.py`).

---

## 6. Conventions that cause mistakes

- **θ on the hole** is measured at the hole centre from the +x axis, counter-clockwise, in degrees,
  range −180 … 180. θ = ±90° is the top/bottom of the hole (the classic peak under x-tension),
  θ = 0°/180° the right/left side.
- **Hole centre** = (150 + ex, 50 + ey) mm.
- **Sign:** tension positive. λ = −1 means σy = −100 MPa (compression).
- **Meshes differ between cases** — node and element IDs cannot be matched across cases. For
  image-type models interpolate each case onto a common grid; for point-wise models use (x, y).

---

## 7. How we know the data is right (`validation/`)

| Check | Result |
|---|---|
| Mesh convergence (D/W = 0.3, 4.1° → 0.5° hole elements) | Kt changes 0.03 % between the two finest meshes; 1° mesh within 0.04 % of extrapolated value |
| Kirsch analytical solution (small hole, λ = 0 / 1 / −1) | Kt 3.008 / 2.007 / 4.021 vs 3 / 2 / 4; whole hole-edge curve within 0.9 MPa |
| Heywood / Peterson finite-width formulas | within 0.6–0.8 % (the formulas' own accuracy) |
| Independent open-source FE solver (gmsh + scikit-fem), 15 random cases | max difference 0.09 % in Kt |
| Thin-ligament cases vs fine-mesh reference | within 0.2 % |
| Per-case checks, all 380 | no failures; reactions ≈ 1e-8 N; free-edge radial stress ≤ 0.18 % of peak |

---

## 8. Things you should know before modelling

- **5 cases** (all near-equibiaxial, large hole close to the top edge) have the plate maximum on the
  top plate edge above the hole, up to 15 % higher than the hole maximum. `kt` is always the
  hole-edge maximum; `kt_global` is the plate maximum. Both are physical (confirmed independently).
- **88 cases** have two almost equal peaks on opposite sides of the hole (`second_peak_ratio ≥ 0.99`).
  For them the peak *angle* is ill-defined — evaluate location models by the true stress at the
  predicted location, not by the angle difference.
- **2 test cases** have Kt above the training maximum (12.82 and 12.62 vs 11.85).
- Kt ranges 2.1 – 12.8 and errors are relative, so training on log(Kt) works better than on Kt.

---

## 9. Reference results (to compare against)

Mean / max absolute % error of Kt on the test sets, models trained only on `train`:

| Model | test_interp | test_extrap |
|---|---|---|
| Analytical (Kirsch × Heywood, ignores hole offset) | 11.8 / 48.8 | 19.4 / 44.3 |
| Linear regression (10 features) | 7.6 / 28.9 | 12.5 / 29.6 |
| Random forest (tuned) | 3.6 / 21.0 | 10.5 / 25.0 |
| LightGBM (tuned) | 1.9 / 14.4 | 6.0 / 17.1 |
| MLP (tuned) | 1.0 / 7.9 | 2.7 / 11.4 |
| Gaussian process, 4 raw inputs (`quickstart.py`) | 0.4 / 2.7 | 1.5 / 5.7 |
| Gaussian process, 10 features | **0.17 / 1.3** | **0.58 / 2.9** |

Features used: `dw, exw, eyw, lam, lig_top_w, lig_bot_w, lig_min_w, net_w, dist_end_w, kt_kirsch`.

---

## 10. Regenerating or extending the data (`simulation/`)

| File | Purpose |
|---|---|
| `plate_hole_2d_single_case.mac` | one case, edit the inputs at the top, run in Mechanical APDL (File → Read Input From) |
| `plate_case_batch.mac` | one dataset case; called by the driver below |
| `run_dataset.mac` | driver that ran all 380 cases (put it next to `plate_case_batch.mac`, read it, ~20–30 min) |
| `doe.py` | generates parameter tables and drivers: `python doe.py dataset` |
| `doe_*.csv` | the exact parameters of every case |
| `ansys_scalars_raw_output.csv` | raw per-case summary written by ANSYS |

ANSYS writes flat files `nod_XXXX.csv`, `hol_XXXX.csv`, `ele_XXXX.csv` and `scalars.csv`; this
package was built from those.

---

## 11. Version and contact

- **v1** — 2026-10-18 — 380 cases, isotropic steel. Next: composite laminates (per-ply stresses,
  failure indices) in the same format.
- Contact: Ege (IUE Aerospace Engineering) — ask before publishing results based on this data.
