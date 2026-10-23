/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef CHIRELAXATIONPARAMETERS_HPP_
#define CHIRELAXATIONPARAMETERS_HPP_

#include "EkpyroticInitialData.hpp"
#include "GRParmParse.hpp"
#include "Potential.hpp"

#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <cmath>
#include <string>

//! Parameters for the initial relaxation of chi (read from "relaxation.*")
/*!
    For time < relaxation.time only chi (and, for the hyperbolic method, the
    auxiliary variable theta) is evolved. relaxation.method selects

    - "parabolic" (default): rhs_chi = speed * Ham / chi. On a conformally
      flat slice Ham contains 2 Laplacian(chi), so this is a diffusion
      equation and stability with dt = dt_multiplier * dx requires
      speed <~ 0.087 chi_min dx / dt_multiplier for RK4 with fourth-order
      derivatives. The default is 0.05 dx / dt_multiplier.

    - "hyperbolic": rhs_chi = theta, rhs_theta = A * Ham / chi - eta * theta,
      a damped wave equation with wave speed sqrt(2 A). Stability requires
      A <~ 0.245 / dt_multiplier^2 (independent of dx); the default is
      0.1 / dt_multiplier^2. The slowest (k = 0) mode is critically damped for
      eta = 2 sqrt(A * 3 kappa^2 Pi0^2), which is printed by check_params().
      eta must be set explicitly: without damping the relaxation oscillates
      and never converges.
*/
struct chi_relaxation_params_t
{
    amrex::Real time{0.0};
    amrex::Real speed{0.0};
    std::string method{"parabolic"}; //!< "parabolic" or "hyperbolic"
    amrex::Real coefficient_A{0.0};  //!< hyperbolic: A in rhs_theta
    amrex::Real eta_damping{0.0};    //!< hyperbolic: damping eta

    static constexpr amrex::Real default_speed_factor = 0.05;
    static constexpr amrex::Real max_speed_factor     = 0.08;
    static constexpr amrex::Real default_A_factor     = 0.1;
    static constexpr amrex::Real max_A_factor         = 0.2;

    [[nodiscard]] bool is_hyperbolic() const { return method == "hyperbolic"; }

    static amrex::Real coarsest_dx()
    {
        GRParmParse geometry_pp("geometry");
        amrex::Real dx{};
        geometry_pp.get("coarsest_dx", dx);
        return dx;
    }

    static amrex::Real dt_multiplier()
    {
        GRParmParse evolution_pp("evolution");
        amrex::Real dt_mult{};
        evolution_pp.get("dt_multiplier", dt_mult);
        return dt_mult;
    }

    static amrex::Real dx_over_dt_multiplier()
    {
        return coarsest_dx() / dt_multiplier();
    }

    //! eta that critically damps the k = 0 mode, 2 sqrt(A * 3 kappa^2 Pi0^2)
    static amrex::Real critical_eta(amrex::Real a_coefficient_A)
    {
        EkpyroticInitialData::params_t initial_data_params;
        initial_data_params.fill_params();
        const Potential potential;
        const auto background = EkpyroticInitialData::compute_background(
            initial_data_params, potential);
        return 2.0 * std::sqrt(a_coefficient_A * 3.0 * background.kappa2 *
                               background.Pi0 * background.Pi0);
    }

    static void check_params()
    {
        GRParmParse pp("relaxation");

        amrex::Real time{0.0};
        pp.queryAdd("time", time);
        if (time < 0.0)
        {
            pp.error("time", "must be >= 0.0");
        }

        std::string method{"parabolic"};
        pp.queryAdd("method", method);
        if (method != "parabolic" && method != "hyperbolic")
        {
            pp.error("method", "must be \"parabolic\" or \"hyperbolic\"");
        }
        const bool relaxing   = (time > 0.0);
        const bool hyperbolic = (method == "hyperbolic");

        // Parabolic relaxation
        const amrex::Real dx_over_dtm = dx_over_dt_multiplier();
        amrex::Real speed             = default_speed_factor * dx_over_dtm;
        pp.queryAdd("speed", speed);
        if (relaxing && !hyperbolic && speed <= 0.0)
        {
            pp.error("speed", "must be > 0.0 when relaxation.time > 0");
        }
        if (!hyperbolic && speed > max_speed_factor * dx_over_dtm)
        {
            pp.warning("speed",
                       "is larger than ~0.08 * coarsest_dx / dt_multiplier; "
                       "the chi relaxation (a diffusion equation) is likely "
                       "to be numerically unstable");
        }

        // Hyperbolic relaxation
        const amrex::Real dtm = dt_multiplier();
        amrex::Real coefficient_A{default_A_factor / (dtm * dtm)};
        pp.queryAdd("A", coefficient_A);
        amrex::Real eta_damping{0.0};
        pp.queryAdd("eta", eta_damping);
        if (eta_damping < 0.0)
        {
            pp.error("eta", "must be >= 0.0");
        }
        if (relaxing && hyperbolic)
        {
            if (coefficient_A <= 0.0)
            {
                pp.error("A", "must be > 0.0 for the hyperbolic relaxation");
            }
            if (coefficient_A > max_A_factor / (dtm * dtm))
            {
                pp.warning("A", "is larger than ~0.2 / dt_multiplier^2; the "
                                "hyperbolic relaxation is likely to be "
                                "numerically unstable");
            }
            if (eta_damping == 0.0)
            {
                pp.error("eta", "must be > 0.0 for the hyperbolic relaxation: "
                                "without damping it oscillates and never "
                                "converges");
            }
            const amrex::Real dt = dtm * coarsest_dx();
            if (eta_damping * dt > 1.0)
            {
                pp.warning("eta", "eta * dt > 1; the damping term is likely "
                                  "to be numerically unstable");
            }
            amrex::Print() << "chi_relaxation_params_t: hyperbolic relaxation "
                              "with A = "
                           << coefficient_A << ", eta = " << eta_damping
                           << " (critical damping of the k = 0 mode: eta = "
                           << critical_eta(coefficient_A) << ")\n";
        }
    }

    void fill_params()
    {
        GRParmParse pp("relaxation");
        pp.get("time", time);
        pp.get("speed", speed);
        pp.get("method", method);
        pp.get("A", coefficient_A);
        pp.get("eta", eta_damping);
    }
};

#endif /* CHIRELAXATIONPARAMETERS_HPP_ */
