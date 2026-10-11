/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef CHIRELAXATIONPARAMETERS_HPP_
#define CHIRELAXATIONPARAMETERS_HPP_

#include "GRParmParse.hpp"

#include <AMReX_REAL.H>

//! Parameters for the initial relaxation of chi.
/*!
    For time < relaxation.time all RHS are zero apart from
    rhs_chi = relaxation.speed * Ham. On a conformally flat slice Ham contains
    2 Laplacian(chi), so this is a diffusion equation and stability with
    dt = dt_multiplier * dx requires speed <~ 0.087 dx / dt_multiplier for
    RK4 with fourth-order derivatives. The default is 0.05 dx / dt_multiplier.
*/
struct chi_relaxation_params_t
{
    amrex::Real time{0.0};
    amrex::Real speed{0.0};

    static constexpr amrex::Real default_speed_factor = 0.05;
    static constexpr amrex::Real max_speed_factor     = 0.08;

    static amrex::Real dx_over_dt_multiplier()
    {
        GRParmParse geometry_pp("geometry");
        amrex::Real coarsest_dx{};
        geometry_pp.get("coarsest_dx", coarsest_dx);

        GRParmParse evolution_pp("evolution");
        amrex::Real dt_multiplier{};
        evolution_pp.get("dt_multiplier", dt_multiplier);

        return coarsest_dx / dt_multiplier;
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

        const amrex::Real dx_over_dtm = dx_over_dt_multiplier();
        amrex::Real speed             = default_speed_factor * dx_over_dtm;
        pp.queryAdd("speed", speed);
        if (time > 0.0 && speed <= 0.0)
        {
            pp.error("speed", "must be > 0.0 when relaxation.time > 0");
        }
        if (speed > max_speed_factor * dx_over_dtm)
        {
            pp.warning("speed",
                       "is larger than ~0.08 * coarsest_dx / dt_multiplier; "
                       "the chi relaxation (a diffusion equation) is likely "
                       "to be numerically unstable");
        }
    }

    void fill_params()
    {
        GRParmParse pp("relaxation");
        pp.get("time", time);
        pp.get("speed", speed);
    }
};

#endif /* CHIRELAXATIONPARAMETERS_HPP_ */
