/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef POTENTIAL_HPP_
#define POTENTIAL_HPP_

#include "GRParmParse.hpp"
#include "ScalarFieldVars.hpp"

#include <AMReX_GpuQualifiers.H>
#include <AMReX_REAL.H>

#include <cmath>

/// Exponential (ekpyrotic) potential V(phi) = V0 * exp(-c * phi)
class Potential
{
  public:
    struct params_t
    {
        amrex::Real V0{-1.0};
        amrex::Real c{10.0};

        static void check_params()
        {
            GRParmParse scalar_field_pp("scalar_field");
            amrex::Real V0_value{-1.0};
            amrex::Real c_value{10.0};
            scalar_field_pp.queryAdd("V0", V0_value);
            scalar_field_pp.queryAdd("c", c_value);
            if (c_value <= 0.0)
            {
                scalar_field_pp.error("c", "must be > 0.0");
            }
        }

        void fill_params()
        {
            GRParmParse scalar_field_pp("scalar_field");
            scalar_field_pp.get("V0", V0);
            scalar_field_pp.get("c", c);
        }
    };

    Potential() { m_params.fill_params(); }

    AMREX_GPU_HOST_DEVICE
    AMREX_FORCE_INLINE explicit Potential(params_t a_params)
        : m_params(a_params)
    {
    }

    /// The potential as a function of phi (also used for the initial data)
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    V_of_phi(amrex::Real a_phi) const
    {
        return m_params.V0 * std::exp(-m_params.c * a_phi);
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    compute_potential(amrex::Real &V_of_phi_out, amrex::Real &dVdphi,
                      const ScalarFieldVars &vars) const
    {
        V_of_phi_out = V_of_phi(vars.phi());
        dVdphi       = -m_params.c * V_of_phi_out;
    }
    // NOLINTEND(bugprone-easily-swappable-parameters)

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE const params_t &
    get_params() const
    {
        return m_params;
    }

  private:
    params_t m_params{};
};

#endif /* POTENTIAL_HPP_ */
