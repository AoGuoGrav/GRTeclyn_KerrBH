/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef XCTSPARAMETERS_HPP_
#define XCTSPARAMETERS_HPP_

#include "EkpyroticBHData.hpp"
#include "GRParmParse.hpp"
#include "Potential.hpp"
#include "XCTSRelaxation.hpp"

#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <cmath>
#include <string>

//! Parameters of the XCTS relaxation and of the hand-over to the evolution
//! (read from "xcts.*")
/*!
    For time < xcts.relaxation_time only the XCTS unknowns u, v, beta^i and
    their pseudo-time derivatives evolve (see XCTSRelaxation); the BSSN
    variables are recomputed from them after every step.

    Stability (RK4, fourth-order derivatives, dt = dt_multiplier dx) needs
    A_u, A_v <~ 0.45 / dt_multiplier^2 and A_beta <~ 0.35 / dt_multiplier^2
    (the vector Laplacian has wave speed sqrt(4 A / 3)). The default damping
    rates critically damp the slowest mode of each field: the k = 0 mode of u
    and v (masses 3/2 kappa^2 Pi0^2 and dtK) and the longest periodic mode
    of beta (k = 2 pi / L_max).

    xcts.lapse = "xcts" solves the lapse equation; "unit" sets alpha = 1
    (CTS with alpha_tilde = psi^-6). At the end of the relaxation the lapse
    is either kept (evolution_lapse = "xcts") or replaced by psi^-2
    ("precollapsed"), and the shift kept or set to zero (evolution_shift).
*/
struct xcts_params_t
{
    amrex::Real relaxation_time{0.0};
    std::string lapse{"xcts"};
    std::string evolution_lapse{"precollapsed"};
    std::string evolution_shift{"xcts"};
    XCTSRelaxation::coefficients_t coefficients{};
    amrex::Real diagnostic_excision_radius{0.0};

    static constexpr amrex::Real default_A_factor      = 0.2;
    static constexpr amrex::Real default_A_beta_factor = 0.15;
    static constexpr amrex::Real max_A_factor          = 0.45;
    static constexpr amrex::Real max_A_beta_factor     = 0.35;

    [[nodiscard]] bool solve_lapse() const { return lapse == "xcts"; }

    static amrex::Real dt_multiplier()
    {
        GRParmParse evolution_pp("evolution");
        amrex::Real dt_mult{};
        evolution_pp.get("dt_multiplier", dt_mult);
        return dt_mult;
    }

    static void check_params()
    {
        GRParmParse pp("xcts");

        amrex::Real relaxation_time{0.0};
        pp.queryAdd("relaxation_time", relaxation_time);
        if (relaxation_time < 0.0)
        {
            pp.error("relaxation_time", "must be >= 0.0");
        }

        std::string lapse{"xcts"};
        pp.queryAdd("lapse", lapse);
        if (lapse != "xcts" && lapse != "unit")
        {
            pp.error("lapse", "must be \"xcts\" or \"unit\"");
        }
        std::string evolution_lapse{"precollapsed"};
        pp.queryAdd("evolution_lapse", evolution_lapse);
        if (evolution_lapse != "xcts" && evolution_lapse != "precollapsed")
        {
            pp.error("evolution_lapse", "must be \"xcts\" or \"precollapsed\"");
        }
        std::string evolution_shift{"xcts"};
        pp.queryAdd("evolution_shift", evolution_shift);
        if (evolution_shift != "xcts" && evolution_shift != "zero")
        {
            pp.error("evolution_shift", "must be \"xcts\" or \"zero\"");
        }

        const amrex::Real dtm  = dt_multiplier();
        const amrex::Real dtm2 = dtm * dtm;
        amrex::Real A_u{default_A_factor / dtm2};
        amrex::Real A_v{default_A_factor / dtm2};
        amrex::Real A_beta{default_A_beta_factor / dtm2};
        pp.queryAdd("A_u", A_u);
        pp.queryAdd("A_v", A_v);
        pp.queryAdd("A_beta", A_beta);
        if (A_u <= 0.0 || A_v <= 0.0 || A_beta <= 0.0)
        {
            pp.error("A_u", "A_u, A_v and A_beta must be > 0.0");
        }
        if (A_u > max_A_factor / dtm2 || A_v > max_A_factor / dtm2)
        {
            pp.warning("A_u", "A_u or A_v is larger than ~0.45 / "
                              "dt_multiplier^2; the relaxation is likely to "
                              "be numerically unstable");
        }
        if (A_beta > max_A_beta_factor / dtm2)
        {
            pp.warning("A_beta", "is larger than ~0.35 / dt_multiplier^2; the "
                                 "relaxation is likely to be numerically "
                                 "unstable");
        }

        // Critical damping of the slowest mode of each field
        EkpyroticBHData::bh_params_t bh;
        bh.fill_params();
        EkpyroticBHData::cosmo_params_t cosmo;
        cosmo.fill_params();
        const Potential potential;
        const auto derived =
            EkpyroticBHData::compute_derived(bh, cosmo, potential);
        const amrex::Real mass_u2 =
            1.5 * derived.kappa2 * derived.Pi0 * derived.Pi0;
        const amrex::Real mass_v2 = std::max(derived.dtK, 0.0);
        const amrex::Real k_min =
            2.0 * M_PI /
            *std::max_element(derived.domain_lengths.begin(),
                              derived.domain_lengths.end());
        amrex::Real eta_u{2.0 * std::sqrt(A_u * mass_u2)};
        amrex::Real eta_v{2.0 * std::sqrt(A_v * mass_v2)};
        amrex::Real eta_beta{2.0 * std::sqrt(4.0 / 3.0 * A_beta) * k_min};
        pp.queryAdd("eta_u", eta_u);
        pp.queryAdd("eta_v", eta_v);
        pp.queryAdd("eta_beta", eta_beta);
        if (eta_u <= 0.0 || eta_v <= 0.0 || eta_beta <= 0.0)
        {
            pp.error("eta_u", "eta_u, eta_v and eta_beta must be > 0.0: "
                              "without damping the relaxation never "
                              "converges");
        }

        amrex::Real diagnostic_excision_radius{bh.mass};
        pp.queryAdd("diagnostic_excision_radius", diagnostic_excision_radius);

        if (relaxation_time > 0.0)
        {
            amrex::Print() << "xcts_params_t: lapse = " << lapse
                           << ", A_u = " << A_u << ", A_v = " << A_v
                           << ", A_beta = " << A_beta << ", eta_u = " << eta_u
                           << ", eta_v = " << eta_v
                           << ", eta_beta = " << eta_beta << "\n";
        }
    }

    void fill_params()
    {
        GRParmParse pp("xcts");
        pp.get("relaxation_time", relaxation_time);
        pp.get("lapse", lapse);
        pp.get("evolution_lapse", evolution_lapse);
        pp.get("evolution_shift", evolution_shift);
        pp.get("A_u", coefficients.A_u);
        pp.get("A_v", coefficients.A_v);
        pp.get("A_beta", coefficients.A_beta);
        pp.get("eta_u", coefficients.eta_u);
        pp.get("eta_v", coefficients.eta_v);
        pp.get("eta_beta", coefficients.eta_beta);
        pp.get("diagnostic_excision_radius", diagnostic_excision_radius);
        coefficients.solve_lapse = solve_lapse();
    }
};

#endif /* XCTSPARAMETERS_HPP_ */
