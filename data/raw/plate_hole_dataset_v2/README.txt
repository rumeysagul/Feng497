PLATE-WITH-A-HOLE STRESS DATASET - v2  (steel, linear elastic, ANSYS)
=====================================================================

535 finite-element simulations of a 2D steel plate with one circular hole, made in
ANSYS Mechanical APDL 2023 R1 for the IUE thesis project "Predicting cracks in composite
structures with machine learning" (stage 1: isotropic steel, von Mises stress).
Every case contains the full nodal stress field, the mesh, and the stresses around the hole.

v2 = v1 (380 cases, hole diameter 10-65 mm) + 155 bigger-hole cases (60-80 mm).
All cases were run with the same model, mesh rules and loads - one consistent dataset.

GOAL FOR ML: predict the stress (especially the peak at the hole edge, Kt) from 4 design
parameters, without running FE.


1. FOLDER CONTENTS
------------------

  README.txt                          this file
  data/cases.csv                      ONE ROW PER CASE: inputs, derived features, targets
                                      (Kt, peak angle), train/test split. Start here.  (124 kB)
  data/hole_profiles.npz              stresses on the hole edge, 360 points per case
                                      (1 degree grid), all cases in one array          (7 MB)
  data/hole_profiles_long.csv         same as above in long CSV form (Excel / MATLAB)  (10 MB)
  data/fields/nodes_all.parquet       FULL STRESS FIELD: every corner node of every
                                      case (4.78 million rows)                         (159 MB)
  data/fields/elements_all.parquet    mesh connectivity of every case (4.60 M rows)    (77 MB)
  validation/                         evidence that the FE data is correct (section 7)

  data/fields/ is only in the full version of the package. Without it you can still
  predict Kt and the hole-edge stresses; you need it to predict the stress everywhere.

Software: Python with  pip install pandas numpy scikit-learn pyarrow matplotlib
          MATLAB: readtable for CSV, parquetread for Parquet (R2019a or newer)


2. LOADING THE DATA (Python)
----------------------------

    import numpy as np, pandas as pd

    cases = pd.read_csv("data/cases.csv")                 # one row per case
    train = cases[cases.split == "train"]                 # 425 cases - train and tune on these only
    test  = cases[cases.split != "train"]                 # 110 cases - final evaluation only

    prof = np.load("data/hole_profiles.npz")              # hole edge, 1 degree grid
    vm_edge = prof["seqv"]                                # shape (535, 360), rows ordered as prof["case"]

    # full field of ONE case (reading all 4.78 M rows at once also works, ~1 GB RAM)
    nodes = pd.read_parquet("data/fields/nodes_all.parquet", filters=[("case", "==", 3121)])
    elems = pd.read_parquet("data/fields/elements_all.parquet", filters=[("case", "==", 3121)])

A first model for Kt (Gaussian process on the 4 raw inputs, about 0.6 % mean error on test_interp):

    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = ["dw", "exw", "eyw", "lam"]
    k = C(1.0) * Matern(length_scale=np.ones(4), nu=2.5) + WhiteKernel(1e-6, (1e-10, 1e-2))
    m = make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=3))
    m.fit(train[X], np.log(train.kt))                     # train on log(Kt): errors are relative
    t = cases[cases.split == "test_interp"]
    err = 100 * np.abs(np.exp(m.predict(t[X])) / t.kt - 1)
    print(err.mean(), err.max())


3. THE PHYSICAL MODEL (same for all cases)
------------------------------------------

           y           top edge:   sigma_y = lambda * sigma_x
          100 +------------------------------------------+
              |                                          |
   sigma_x <--|               (o)  hole                  |--> sigma_x     plate 300 x 100 x 2 mm
              |          centre (150+ex, 50+ey)          |
            0 +------------------------------------------+
              0  A        bottom edge: sigma_y = lambda*sigma_x   B  300  --> x

  Units          mm, N, MPa
  Coordinates    origin at bottom-left corner A; x along the 300 mm length, y along the 100 mm width
  Plate          L = 300 mm, W = 100 mm, thickness t = 2 mm
  Material       steel, linear elastic isotropic: E = 200 GPa, nu = 0.3 (no plasticity, no yield)
  Analysis       2D static, plane stress, PLANE183 (8-node quadratic)
  Loads          uniform edge traction: sigma_x = 100 MPa tension on left and right edges;
                 sigma_y = lambda * 100 MPa on top and bottom edges (lambda < 0 -> compression)
  Supports       rigid-body only: corner A (0,0) UX = UY = 0, corner B (300,0) UY = 0.
                 Loads are self-balanced, reactions ~ 1e-8 N
  Mesh           1 degree elements on the hole edge (360 around), 3 mm far field,
                 top/bottom edges refined to ligament/6

WHY EVERY STRESS IS "PER 100 MPa": the model is linear and loaded by tractions, so all stresses
scale exactly with load, and E and nu do not affect them. Divide any stress by 100 to get the
dimensionless stress-concentration value (columns ending in _n, and kt); multiply by your real
nominal stress to get MPa. Valid only while the material stays elastic - most cases exceed the
yield stress of real steel at 100 MPa nominal load.


4. THE FOUR VARIED PARAMETERS AND THE SPLITS
--------------------------------------------

  Parameter  Column(s)        Meaning                                       Range
  D          D [mm], dw=D/W   hole diameter                                 10 - 80 mm (D/W 0.10 - 0.80)
  ex         ex [mm], exw     hole x-offset from plate centre (+ = right)   -50 ... +50 mm
  ey         ey [mm], eyw     hole y-offset from plate centre (+ = up)      up to +-20 mm, each ligament >= 10 mm
  lambda     lam              load ratio sigma_y / sigma_x                  -1 (pure shear) ... 0 (uniaxial) ... +1 (equibiaxial)

The LIGAMENT is the distance from the hole edge to the top or bottom plate edge = 50 - D/2 - |ey|.
It is never below 10 mm, so the allowed |ey| shrinks as the hole grows:
D = 60 mm -> |ey| <= 10 mm;  D = 80 mm -> the hole must be centred (ey = 0). 80 mm is the largest hole.

Sampled with Latin hypercube sampling (separate seeds per set, no duplicates between sets):

  split        Cases  Case IDs                 Hole size   Use it for
  train        425    1-300 and 3001-3125      10-80 mm    training AND all tuning (cross-validation inside these 425)
  test_interp   50    1001-1050                10-60 mm    final test
  test_mid      30    2001-2030                60-65 mm    final test
  test_big      30    4001-4030                60-80 mm    final test, bigger holes

Cases 3121-3125 are deliberate corner cases (largest hole with 10 mm ligaments on both sides,
largest hole at the far left and right, 60 and 70 mm holes at their maximum offset).

RULES SO RESULTS ARE COMPARABLE AND HONEST
  1. Never train or tune on any test_* set. Look at them once, at the end.
  2. When you use the field data (one row per node), split BY CASE, never by node - nodes of the
     same plate are near-duplicates and a node-level split gives fake, inflated scores.
  3. Report errors in physical terms (% error of Kt, MPa), not only R^2. Most of the plate sits
     near the nominal stress, so field-level R^2 looks excellent even when the peak is wrong.
  4. Report the three test sets separately - accuracy for big holes differs from small ones.


5. COLUMN DICTIONARY
--------------------

data/cases.csv - one row per case

  case               -     case ID (key in every file)
  split              -     train, test_interp, test_mid, test_big (section 4)
  split_v1           -     set name when the case was generated
                           (train, test_interp, test_extrap, train_big, test_interp_big)
  origin             -     v1 (cases 1-2030) or v2_extension (cases 3001-4030)
  D, ex, ey          mm    hole diameter, x-offset, y-offset (section 4)
  lam                -     load ratio sigma_y / sigma_x
  dw, exw, eyw       -     D/W, ex/W, ey/W
  lig_top_w          -     ligament hole edge -> top plate edge, divided by W
  lig_bot_w          -     ligament hole edge -> bottom plate edge, divided by W
  lig_min_w          -     smaller of the two ligaments / W
  lig_ratio          -     smaller ligament / larger ligament (1 = hole centred in y)
  net_w              -     net-section width (W - D) / W
  dist_end_w         -     distance hole edge -> nearest loaded x-end / W
  kt_kirsch          -     analytical infinite-plate peak for this lambda: max(|3 - lam|, |3 lam - 1|)
                           (a feature, not a target)
  kt                 -     MAIN TARGET: max von Mises on the hole edge / sigma_x
  theta_pk           deg   angle of that peak on the hole (convention in section 6)
  kt_global          -     max von Mises anywhere in the plate / sigma_x (differs from kt in 26 cases, section 8)
  peak_on_hole       bool  True if the plate maximum is on the hole edge
  second_peak_ratio  -     highest hole-edge value more than 45 deg away from the peak, divided by the peak
                           (>= 0.99 means two near-equal peaks)
  n_nodes, n_elem    -     mesh size of this case (all nodes incl. midside)
  mesh_err_pct       %     ANSYS energy-norm error estimate (all <= 0.14 %)

data/hole_profiles.npz - hole edge, 1 degree grid

  case        (535,)       case IDs, same order as the rows of the arrays (and of cases.csv)
  theta_deg   (360,)       -180 ... 179
  seqv        (535, 360)   von Mises / sigma_x
  stt         (535, 360)   hoop (tangential) stress / sigma_x - the only non-zero stress on a free hole edge
  sx, sy, sxy (535, 360)   global stress components / sigma_x

  hole_profiles_long.csv has the same data with columns: case, theta_deg, seqv_n, stt_n, sx_n, sy_n, sxy_n

data/fields/nodes_all.parquet - full field

  case               -     case ID
  node_id            -     ANSYS node number (UNIQUE ONLY WITHIN A CASE; every case has its own mesh)
  x, y               mm    node position (origin bottom-left corner)
  sx, sy, sxy        MPa   stress components, global x-y axes
  s1, s2             MPa   in-plane principal stresses
  seqv               MPa   von Mises stress

  Corner nodes only (midside nodes of the quadratic elements are not exported). Stresses are ANSYS
  averaged nodal values. Applied sigma_x is 100 MPa in every case.

data/fields/elements_all.parquet - mesh

  case, elem_id, n1 ... n8:  n1-n4 are the corner nodes (counter-clockwise), n5-n8 midside nodes
  (not in nodes_all). A few elements are triangles: then n3 == n4. To draw or interpolate the
  mesh use n1-n4 only, e.g. with matplotlib PolyCollection.


6. CONVENTIONS THAT CAUSE MISTAKES
----------------------------------

  - THETA on the hole is measured at the hole centre from the +x axis, counter-clockwise, in degrees,
    range -180 ... 180. theta = +-90 is the top/bottom of the hole (the classic peak under x-tension),
    theta = 0 / 180 the right / left side.
  - HOLE CENTRE = (150 + ex, 50 + ey) mm.
  - SIGN: tension positive. lambda = -1 means sigma_y = -100 MPa (compression).
  - MESHES DIFFER BETWEEN CASES - node and element IDs cannot be matched across cases. For
    image-type models interpolate each case onto a common grid; for point-wise models use (x, y).


7. HOW WE KNOW THE DATA IS RIGHT (validation/)
----------------------------------------------

  Check                                                  Result
  Mesh convergence (D/W = 0.3, 4.1 -> 0.5 deg elements)  Kt changes 0.03 % between the two finest meshes;
                                                         1 deg mesh within 0.04 % of the extrapolated value
  Kirsch analytical solution (small hole,                Kt 3.008 / 2.007 / 4.021 vs 3 / 2 / 4;
    lambda = 0 / 1 / -1)                                 whole hole-edge curve within 0.9 MPa
  Heywood / Peterson finite-width formulas               within 0.6-0.8 % (the formulas' own accuracy)
  Independent open-source FE solver (gmsh + scikit-fem),  max difference 0.09 % in Kt
    15 random v1 cases
  Thin-ligament cases vs fine-mesh reference             within 0.2 %
  v2 corner cases 3121-3125 vs independent solver        Kt within 0.16 %, plate maximum within 0.18 %
  Per-case checks, all 535                               no failures; mesh error <= 0.14 %; reactions ~ 1e-8 N;
                                                         free-edge radial stress <= 0.18 % of peak

  Files in validation/:
    mesh_convergence_and_kirsch.csv               mesh convergence and Kirsch runs
    figures/fig_mesh_convergence.png, fig_kirsch.png
    pilot2_comparison.csv                         thin-ligament cases before / after the mesh fix
    spot_check_dataset.csv                        15 dataset cases re-solved with the independent solver
    big_hole_corner_cases_vs_independent.csv      v2 corner cases vs the independent solver
    quality_report_all_cases.csv                  automated checks for every case
    independent_fe.py                             the independent solver (gmsh + scikit-fem)


8. THINGS YOU SHOULD KNOW BEFORE MODELLING
------------------------------------------

  - 26 cases (all near-equibiaxial, lambda >= 0.7, with large holes) have the plate maximum on the
    top or bottom plate edge, up to 35 % higher than the hole maximum (case 3122: 759 MPa on the
    edge vs 562 MPa on the hole). kt is always the hole-edge maximum; kt_global is the plate maximum.
    Both are physical (confirmed with the independent solver).
  - 114 cases have two almost equal peaks on opposite sides of the hole (second_peak_ratio >= 0.99).
    For them the peak ANGLE is ill-defined - evaluate location models by the true stress at the
    predicted location, not by the angle difference.
  - Kt ranges 2.1 - 18.6. The highest values come from large holes with 10 mm ligaments under
    compression-dominated loading (lambda < 0). Train on log(Kt).
  - Every test case lies inside the training range of the inputs; there is no extrapolation test
    set (no larger hole is possible with 10 mm ligaments).


9. REFERENCE RESULTS (to compare against)
-----------------------------------------

Models trained on the 425 training cases, evaluated on the three test sets:

                                                   test_interp      test_mid        test_big
                                                   (10-60 mm)       (60-65 mm)      (60-80 mm)
  Analytical Kt (Kirsch x Heywood), mean / max     11.8 / 48.8 %    19.4 / 44.3 %   19.7 / 45.7 %
  Gaussian process Kt, 10 features, mean / max      0.11 / 0.73 %    0.14 / 0.36 %   0.13 / 0.64 %
  Field neural network: von Mises RMS per case       0.17 MPa         0.22 MPa        0.27 MPa
  Field neural network: hole-edge peak, mean / max   0.08 / 0.62 %    0.07 / 0.33 %   0.06 / 0.21 %

  Gaussian process features: dw, exw, eyw, lam, lig_top_w, lig_bot_w, lig_min_w, net_w, dist_end_w, kt_kirsch.
  Field network: one row per node; inputs = the 4 parameters + node position relative to the hole and
  plate edges + the Kirsch infinite-plate stresses at that point; it predicts the correction to the
  Kirsch stresses (4 hidden layers x 256 neurons).


10. VERSION HISTORY
-------------------

  v2  2026-10-24  535 cases. Added 155 cases with D = 60-80 mm (125 train, 30 test). Training set
                  300 -> 425. The v1 set test_extrap was renamed test_mid (its 60-65 mm holes are now
                  inside the training range; it is still never used for training). New columns
                  split_v1 and origin. v1 cases unchanged (verified identical).
  v1  2026-10-18  380 cases, D = 10-65 mm.
