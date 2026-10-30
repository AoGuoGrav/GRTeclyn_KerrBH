# EkpyroticBH

A black hole (puncture) in an ekpyrotic scalar-field cosmology,
`V(phi) = V0 exp(-c phi)`, in a box that is periodic in all directions.
The initial data are found by relaxing the conformally flat XCTS
equations (Hamiltonian constraint, momentum constraint and the lapse
equation from `d_t K`). After that, the CCZ4/BSSN equations with the
scalar field are evolved.

Far from the black hole the cosmological free data are the ones in the
`Ekpyrotic` example (Xue, Garfinkle, Pretorius & Steinhardt,
arXiv:1308.3044), using the same `initial_data.*` parameters. With
`bh.mass = 0` the example reproduces those data exactly.

## Construction

The source files hold the details:

- `EkpyroticBHData.hpp` sets the free data.
- `XCTSRelaxation.hpp` holds the equations and their relaxation.

In outline:

- **Conformal factor.** `psi = 1 + M W(r) / (2 r) + u`. The window `W`
  makes the puncture part periodic, and `u` is solved for.
- **Lapse.** `alpha psi = 1 + M W(r) / (2 r) + v`.
- **Shift.** The shift `beta^i` is stored in `shift1-3`.
- **Mean curvature.** `K = K0` is constant, with
  `K0^2 = 3 kappa^2 <rho> + 24 pi M / V_box`. The black-hole mass enters
  the mean density, which makes the periodic Hamiltonian constraint
  solvable.
- **Matter near the black hole.** For `r < bh.homogeneous_inner` the matter
  is homogeneous with `3 kappa^2 rho = K0^2` (local Friedmann), so
  `psi = 1 + M / (2 r)` solves the Hamiltonian constraint there.
- **Shift weight.** `Ahat = Ahat_free + w L(beta)`, with
  `w = (1 + u)^7 / (2 (1 + v))`. This is exactly XCTS wherever the
  puncture part vanishes. Near the puncture it is a weighted-CTS choice
  that keeps the shift relaxation stable.

## Relaxation and evolution

For `t < xcts.relaxation_time` only `u`, `v` and `beta` relax, using a
damped-wave (hyperbolic) relaxation. All other variables follow from
them.

At every print interval the program writes:

- the XCTS residuals `Ru`, `Rv` and `Rbeta`;
- the BSSN constraints, which are computed independently by finite
  differences.

Both are left out inside `xcts.diagnostic_excision_radius`.

At the end of the relaxation:

- the lapse is replaced by `psi^-2` (`xcts.evolution_lapse = precollapsed`);
- the shift is kept or zeroed (`xcts.evolution_shift`).

Then 1+log slicing and the integrated Gamma-driver take over.

`xcts.lapse = unit` skips the lapse equation (`alpha = 1`, i.e. CTS).

## Parameter files

- `params_test.txt` is a small single-level test with `M = 0.5` in an
  `8^3` box at `64^3` resolution. With this mass the black hole dominates
  the mean density of the box, so treat it as a numerical test, not a
  physical setup.

- `params_amr.txt` uses physical parameters with mesh refinement:
  - box `40^3` (perturbation `mode_number = 2`, wavelength 20) and
    `M = 0.05`, so the black hole is about 1.5% of the mean density. The
    nearest periodic image is causally disconnected during the evolution,
    so this is close to an isolated black hole in FRW;
  - a `64^3` coarse grid plus 8 fixed levels (`FixedGridsTagger` centred
    on the black hole), giving `M / dx = 20` on the finest level.

  It relaxes for `tau = 15`, then writes a checkpoint so the run can be
  restarted into the evolution. It checks that the relaxation works on
  an AMR hierarchy; the diagnostics print one line per level
  (`lev N`). It has not been run yet.
