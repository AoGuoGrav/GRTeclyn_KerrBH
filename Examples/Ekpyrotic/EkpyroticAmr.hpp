/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef EKPYROTICAMR_HPP_
#define EKPYROTICAMR_HPP_

#include "GRAmr.hpp"
#include "ParticleInterpolator.hpp"

//! AMR hierarchy carrying the interpolators used for the x profiles.
class EkpyroticAmr : public GRAmr
{
  public:
    ParticleInterpolator<1> chi_interpolator;
    ParticleInterpolator<1> K_interpolator;
    ParticleInterpolator<1> lapse_interpolator;
    ParticleInterpolator<2> scalar_field_interpolator; // phi, Pi
    ParticleInterpolator<4> constraints_interpolator;  // Ham, Mom1-3
    ParticleInterpolator<3> anisotropy_interpolator;   // sigma2, rho, ratio

    explicit EkpyroticAmr(amrex::LevelBld *a_level_bld) : GRAmr(a_level_bld) {}

    void init(amrex::Real a_start_time, amrex::Real a_stop_time) override
    {
        GRAmr::init(a_start_time, a_stop_time);

        chi_interpolator.setup(this);
        K_interpolator.setup(this);
        lapse_interpolator.setup(this);
        scalar_field_interpolator.setup(this);
        constraints_interpolator.setup(this);
        anisotropy_interpolator.setup(this);
    }
};

#endif /* EKPYROTICAMR_HPP_ */
