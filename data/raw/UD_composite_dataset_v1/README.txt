PLATE-WITH-A-HOLE DATASET - UNIDIRECTIONAL COMPOSITE (UD carbon/epoxy lamina), v1
==================================================================================

808 finite-element simulations of a 2D unidirectional (UD) composite plate with one circular hole,
made in ANSYS Mechanical APDL 2023 R1 for the IUE thesis project "Predicting cracks in composite
structures with machine learning" (stage 2: single UD lamina, continuous fibre angle).
Every case contains the full stress field, the mesh, and the stresses around the hole, plus
first-ply-failure measures (Tsai-Wu and maximum stress) computed from them.

GOAL FOR ML: predict the stresses at the hole and the load at which the first crack (first-ply
failure) starts, and where, from 5 inputs: hole size, hole position (2), load ratio, fibre angle.

Companion dataset: the steel plate-with-hole dataset v2 (same plate, loads and hole positions).


1. FOLDER CONTENTS
------------------

  README.txt                          this file
  data/cases.csv                      ONE ROW PER CASE: inputs, features, targets, split. Start here. (350 kB)
  data/hole_profiles.npz              hole-edge stresses + Tsai-Wu ratio, 360 points per case (1 deg) (18 MB)
  data/hole_profiles_long.csv         same in long CSV form (Excel / MATLAB)                      (24 MB)
  data/fields/nodes_all.parquet       FULL STRESS FIELD, every corner node of every case
                                      (16.6 million rows)                                             (378 MB)
  data/fields/elements_all.parquet    mesh connectivity of every case (16.4 million rows)            (183 MB)
  validation/                         evidence that the FE data is correct (section 8)

  Software: Python with  pip install pandas numpy scikit-learn pyarrow matplotlib
            MATLAB: readtable for CSV, parquetread for Parquet (R2019a or newer)


2. LOADING THE DATA (Python)
----------------------------

    import numpy as np, pandas as pd

    cases = pd.read_csv("data/cases.csv")                 # one row per case
    train = cases[cases.split == "train"]                 # train and tune on these only
    test  = cases[cases.split != "train"]                 # final evaluation only

    prof = np.load("data/hole_profiles.npz")              # hole edge, 1 degree grid
    s22_edge = prof["s22"]                                # shape (cases, 360), rows ordered as prof["case"]

    # full field of ONE case (reading everything at once needs a few GB of RAM)
    nodes = pd.read_parquet("data/fields/nodes_all.parquet", filters=[("case", "==", 5)])
    elems = pd.read_parquet("data/fields/elements_all.parquet", filters=[("case", "==", 5)])

  The fibre angle enters a model best as cos(2*fib) and sin(2*fib) (a lamina at fib and fib+180
  is the same material), not as a raw angle.


3. THE PHYSICAL MODEL (same for all cases)
------------------------------------------

           y           top edge:   sigma_y = lambda * sigma_x
          100 +------------------------------------------+
              |        fibres  ////  at angle fib         |
   sigma_x <--|               (o)  hole                  |--> sigma_x     plate 300 x 100 x 2 mm
              |          centre (150+ex, 50+ey)          |
            0 +------------------------------------------+
              0  A        bottom edge: sigma_y = lambda*sigma_x   B  300  --> x

  Units          mm, N, MPa
  Coordinates    origin at bottom-left corner A; x along the 300 mm length, y along the 100 mm width
  Plate          L = 300 mm, W = 100 mm, thickness t = 2 mm, ONE unidirectional layer
  Material       T300/5208 carbon/epoxy UD lamina, linear elastic orthotropic (axis 1 = fibre, 2 = across)
                   E1 = 181 GPa, E2 = 10.3 GPa, G12 = 7.17 GPa
                   nu12 = 0.28   (major: load along fibres, contraction across)
                   nu21 = nu12*E2/E1 = 0.0159   (minor: load across fibres, contraction along)
                 Strengths (for the failure measures only, not used by the FE solution):
                   XT = XC = 1500 MPa (fibre), YT = 40 MPa, YC = 246 MPa (across), S = 68 MPa (shear)
                 Source: Kaw, Mechanics of Composite Materials, Table 2.1 (values as widely quoted).
  Fibre angle    fib = angle of the fibres from the +x axis, counter-clockwise, 0 ... 90 deg.
                 In ANSYS: element coordinate system rotated by fib (LOCAL + ESYS).
  Analysis       2D static, plane stress, PLANE183 (8-node quadratic)
  Loads          uniform edge traction: sigma_x = 100 MPa tension on left and right edges;
                 sigma_y = lambda * 100 MPa on top and bottom edges (lambda < 0 -> compression)
  Supports       rigid-body only: corner A (0,0) UX = UY = 0, corner B (300,0) UY = 0
  Mesh           ring of width min(R/2, 0.6 x ligament) around the hole, mapped quads, 1 deg around the
                 hole, element thickness at the hole edge 1/5 of its length, growing outwards;
                 3 mm free mesh far away; top/bottom edges refined to ligament/6.
                 (Needed because the stress peak at the hole decays within ~R/20 for fibres
                 along the load - see validation.)

  WHY STRESSES ARE "PER 100 MPa": the FE model is linear, so every stress scales exactly with the
  load. Divide by 100 to get the dimensionless values (columns ending in _n). The failure ratios
  (tw_R, ms_R) are NOT linear in the load; they are defined for this load pattern - see section 7.


4. THE FIVE VARIED PARAMETERS AND THE SPLITS
--------------------------------------------

  Parameter  Column(s)        Meaning                                       Range
  D          D [mm], dw=D/W   hole diameter                                 10 - 80 mm
  ex         ex [mm], exw     hole x-offset from plate centre (+ = right)   -50 ... +50 mm
  ey         ey [mm], eyw     hole y-offset from plate centre (+ = up)      up to +-20 mm, each ligament >= 10 mm
  lambda     lam              load ratio sigma_y / sigma_x                  -1 (pure shear) ... +1 (equibiaxial)
  fib        fib [deg]        fibre angle from +x                           0 ... 90 deg, continuous

  Ligament = distance hole edge -> top/bottom plate edge = 50 - D/2 - |ey| >= 10 mm, so the allowed
  |ey| shrinks as the hole grows (D = 80 mm -> hole centred).
  Angles outside 0-90 are covered by symmetry: fibres at -fib with offset -ey are the mirror image
  (about the plate's mid-line) of fib with +ey; fib + 180 is identical to fib.

  Latin hypercube sampling, separate seeds per set, no duplicates between sets:

  split        Cases  Case IDs     Content                                         Use it for
  train        {n_train:>4}   1-648        600 LHS over all 5 inputs + 20 at fib = 0 +      training AND tuning
                                   20 at fib = 90 + 8 corner cases                 (cross-validate inside)
  test_interp  {n_ti:>4}   2001-2100    100 LHS over all 5 inputs                        final test
  test_0_90    {n_t09:>4}   3001-3060    30 at fib = 0 and 30 at fib = 90 exactly          final test (the two
                                                                                    principal orientations)
  RULES
    1. Never train or tune on test_interp or test_0_90. Use them once, at the end.
    2. Field data (one row per node): split BY CASE, never by node.
    3. Report errors in physical terms (% error of the failure load, MPa), not only R^2.
    4. Report the two test sets separately.


5. COLUMN DICTIONARY
--------------------

data/cases.csv - one row per case (values at the hole edge unless marked "global")

  case, split               case ID (key in every file) and split (section 4)
  D, ex, ey [mm], lam, fib [deg]   the five inputs
  dw, exw, eyw              D/W, ex/W, ey/W
  lig_top_w, lig_bot_w      ligament hole edge -> top / bottom plate edge, divided by W
  lig_min_w                 smaller ligament / W
  net_w                     (W - D) / W
  dist_end_w                hole edge -> nearest loaded x-end, divided by W
  stt_max_n, theta_stt_max  largest TENSILE hoop stress on the hole / sigma_x, and its angle (deg)
  stt_min_n, theta_stt_min  most COMPRESSIVE hoop stress on the hole / sigma_x, and its angle
  s11_max_n, s11_min_n      max / min stress along the fibres on the hole edge / sigma_x
  s22_max_n, s22_min_n      max / min stress across the fibres on the hole edge / sigma_x
  t12_absmax_n              largest |in-plane shear| in fibre axes on the hole edge / sigma_x
  tw_R                      MAIN TARGET: Tsai-Wu strength ratio at the hole = load multiplier at
                            first-ply failure (section 7). < 1 means it already fails at 100 MPa.
  theta_tw                  hole angle where tw_R is reached = where the first crack starts
  fail_load_tw_mpa          100 * tw_R = applied sigma_x [MPa] at first failure (with sigma_y = lam*sigma_x)
  ms_R, theta_ms, ms_mode   same with the maximum-stress criterion, plus the failure mode:
                            fibre_tension, fibre_compression, matrix_tension, matrix_compression, shear
  tw_R_global               smallest Tsai-Wu ratio anywhere in the plate (global)
  tw_global_on_hole         True if that global minimum is on the hole edge
  x_tw_global, y_tw_global  its position [mm]
  n_nodes, n_elem           mesh size (all nodes incl. midside)
  mesh_err_pct              ANSYS energy-norm error estimate [%]

data/hole_profiles.npz - hole edge, 1 degree grid
  case (cases,)  theta_deg (360,) = -180 ... 179
  stt  hoop stress / sigma_x            sx, sy, sxy  global stresses / sigma_x
  s11, s22, t12  fibre-axis stresses / sigma_x
  tw_R           Tsai-Wu strength ratio at each point
  hole_profiles_long.csv: the same, columns case, theta_deg, stt_n, sx_n, sy_n, sxy_n, s11_n, s22_n, t12_n, tw_R

data/fields/nodes_all.parquet - full field (stresses in MPa at sigma_x = 100 MPa)
  case, node_id (unique only within a case), x, y [mm], sx, sy, sxy (global x-y stresses)
  Corner nodes only (midside nodes not exported); ANSYS averaged nodal values.
  Fibre-axis stresses: compute with the formulas in section 6 (see section 9 for code).

data/fields/elements_all.parquet - mesh
  case, elem_id, n1 ... n4: corner nodes, counter-clockwise (the 8-node elements' midside nodes are
  left out). Triangles have n3 == n4. Draw with matplotlib PolyCollection.


6. CONVENTIONS THAT CAUSE MISTAKES
----------------------------------
  - TWO DIFFERENT ANGLES: fib = fibre direction (from +x, CCW); theta (hole angle) = position on the
    hole, measured at the hole centre from +x, CCW, -180 ... 180. theta = +-90 is top/bottom of the hole.
  - Fibre-axis stresses: s11 = sx c^2 + sy s^2 + 2 sxy s c,  s22 = sx s^2 + sy c^2 - 2 sxy s c,
    t12 = (sy - sx) s c + sxy (c^2 - s^2),  with c = cos(fib), s = sin(fib).
  - HOLE CENTRE = (150 + ex, 50 + ey) mm.  SIGN: tension positive.
  - Meshes differ between cases: node/element IDs cannot be matched across cases.
  - Do not use von Mises stress for this material: it is not a meaningful failure measure for composites.


7. FAILURE MEASURES (how tw_R and ms_R are defined)
---------------------------------------------------
  Tsai-Wu:  F1 s11 + F2 s22 + F11 s11^2 + F22 s22^2 + F66 t12^2 + 2 F12 s11 s22 = 1 at failure,
            F1 = 1/XT - 1/XC, F2 = 1/YT - 1/YC, F11 = 1/(XT XC), F22 = 1/(YT YC), F66 = 1/S^2,
            F12 = -0.5 sqrt(F11 F22)  (a common default; F12 is uncertain and affects the result)
  tw_R    = the factor R that, multiplying ALL applied loads (sigma_x and sigma_y together), makes the
            Tsai-Wu index reach 1:  a R^2 + b R - 1 = 0 with a = quadratic terms, b = linear terms.
            fail_load_tw_mpa = 100 * tw_R.
  ms_R    = maximum-stress criterion: smallest of XT/s11, XC/-s11, YT/s22, YC/-s22, S/|t12|.
  These are FIRST-PLY FAILURE (crack initiation) values of a linear-elastic model: they predict where
  and at what load the first matrix crack or fibre failure starts, not the final failure load.
  With other strengths (or a different F12), recompute them from s11, s22, t12 - the stresses
  themselves do not depend on the strengths.


8. HOW WE KNOW THE DATA IS RIGHT (validation/)
----------------------------------------------
  Check                                                     Result
  Material and fibre direction (plate without hole,          eps_x = sigma/E1 and sigma/E2 exactly; measured Poisson ratios
    fibres at 0 and 90 deg)                                   0.2800 and 0.01593 (= nu12 and nu21)
  Mesh convergence (10 mm hole, fibres 0 and 90 deg,         peak changes 0.02 % from 2 deg to 1 deg to 0.5 deg elements
    final ring mesh)                                          (the plain free mesh used for steel was 1-2.5 % too low)
  Independent open-source FE solver (gmsh + scikit-fem,      10 and 30 mm holes, fibres 0/90 deg, lambda 0/-1:
    P2 elements, own orthotropic implementation)              peak hoop stress within 0.17 %
  Infinite-plate theory (Lekhnitskii), 10 mm hole            Kt 6.83 (fibres along load) and 2.41 (across) vs 6.75 and 2.37
                                                              for an infinite plate (difference = finite plate width)
  Independent solver, 6 test cases at fibre angles 8-85 deg  peak hoop stress within 0.25 %, peak angle within 1 deg,
                                                              Tsai-Wu index within 0.06 %
  Independent solver, the 3 cases with the largest ANSYS     peak hoop stress within 0.2 %, Tsai-Wu index within 0.05 %
    error estimate (big holes, near-90 deg fibres)
  Per-case checks, all 808 cases                             no failed runs; energy-norm error estimate median 0.06 %,
                                                              max 0.51 % (13 cases above 0.3 %, all large holes - see row
                                                              above); support reactions <= 1.4e-6 N; stress normal to the
                                                              free hole edge <= 0.009 % of the peak

  Files in validation/:
    mesh_convergence_and_material_checks.csv     ANSYS runs 901-1072 (old free mesh vs final ring mesh)
    spot_check_offaxis_vs_independent.csv        6 test cases, fibre angles 8-85 deg
    spot_check_extreme_vs_independent.csv        3 worst-error-estimate cases
    quality_report_all_cases.csv                 automated checks for every case
    independent_fe_ortho.py                      the independent solver (gmsh + scikit-fem)


9. THINGS YOU SHOULD KNOW BEFORE MODELLING
------------------------------------------
  - FIRST FAILURE HAPPENS AT LOW LOAD: at sigma_x = 100 MPa every case is already past first-ply
    failure (tw_R < 1 for all cases). Median first-failure load 16 MPa (range 2.4 - 90 MPa). Fibres along
    the load fail latest (median 39 MPa for fib < 15 deg), fibres across the load earliest (10 MPa for
    fib > 75 deg). This is matrix cracking (transverse tensile strength is only 40 MPa), not fibre breakage.
  - FAILURE MODE: by the maximum-stress criterion, 647 cases start as matrix_tension and 161 as shear;
    none as fibre failure. Tsai-Wu and maximum-stress agree on the location within 4 deg in 75 % of cases.
  - 14 cases (near-equibiaxial lambda >= 0.74, fibres >= 56 deg, large holes) have their weakest point on
    the top or bottom plate edge, not on the hole (tw_R_global < tw_R). Both are physical.
  - The hoop-stress peaks are much sharper than for steel (|hoop| up to 37 x sigma_x for big holes with
    fibres tangent to the hole under compression-dominated loading). Train on log of stresses / ratios.
  - Fibre angle: use cos(2 fib), sin(2 fib) as inputs (see section 2). fib = 0 and fib = 90 are the edges
    of the sampled range; test_0_90 measures how well a model does exactly there.
  - Every test case lies inside the training ranges; there is no extrapolation test set.
  - Field files hold only the GLOBAL stresses sx, sy, sxy (to keep the download small). Fibre-axis
    stresses for the field: use the formulas in section 6 with the case's fib, e.g.
        c, s = np.cos(np.radians(fib)), np.sin(np.radians(fib))
        nodes["s22"] = nodes.sx*s*s + nodes.sy*c*c - 2*nodes.sxy*s*c


10. REFERENCE RESULTS (simple baseline to compare against)
----------------------------------------------------------
Trained on the 648 training cases, errors on the test sets (no tuning on test data):

  Target: tw_R (first-failure load)               test_interp mean / max    test_0_90 mean / max
    Linear regression (11 features)               20.4 / 131 %              30.2 / 171 %
    Gaussian process, 6 raw inputs                 1.7 / 15 %                3.0 / 18 %
    Gaussian process, 11 features                  0.7 / 5.5 %               1.6 / 9.4 %

  Target: largest |hoop stress| on the hole
    Linear regression (11 features)               16.0 / 58 %               21.2 / 84 %
    Gaussian process, 6 raw inputs                 3.3 / 14 %                3.1 / 15 %
    Gaussian process, 11 features                  1.9 / 14 %                2.1 / 16 %

  6 raw inputs: dw, exw, eyw, lam, cos(2 fib), sin(2 fib).
  11 features: the 6 + lig_top_w, lig_bot_w, lig_min_w, net_w, dist_end_w.
  Gaussian process: Matern 5/2 kernel with one length scale per input + noise term, trained on log(target).
  These are simple baselines - a field model (one row per node) or a model of the whole hole profile
  should do better on the location of the first crack.


11. VERSION
-----------
  v1  2026-11-07  808 cases. Material T300/5208. ANSYS MAPDL 2023 R1, macros plate_ud_case.mac / run_ud_dataset.mac.
  The ANSYS macros and the parameter table are not included in this package; ask Ege if you need to
  regenerate or extend cases (e.g. other materials or laminates).

Contact: Ege (IUE Aerospace Engineering) - ask before publishing results based on this data.
