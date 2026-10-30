/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef EKPYROTICBHAMR_HPP_
#define EKPYROTICBHAMR_HPP_

#include "GRAmr.hpp"
#include "ParticleInterpolator.hpp"

//! AMR hierarchy carrying the interpolators used for the line profiles.
class EkpyroticBHAmr : public GRAmr
{
  public:
    ParticleInterpolator<1> chi_interpolator;
    ParticleInterpolator<1> lapse_interpolator;
    ParticleInterpolator<2> xcts_interpolator;        // xcts_u, xcts_v
    ParticleInterpolator<4> constraints_interpolator; // Ham, Mom1-3

    explicit EkpyroticBHAmr(amrex::LevelBld *a_level_bld) : GRAmr(a_level_bld)
    {
    }

    void init(amrex::Real a_start_time, amrex::Real a_stop_time) override
    {
        GRAmr::init(a_start_time, a_stop_time);

        chi_interpolator.setup(this);
        lapse_interpolator.setup(this);
        xcts_interpolator.setup(this);
        constraints_interpolator.setup(this);
    }
};

#endif /* EKPYROTICBHAMR_HPP_ */
