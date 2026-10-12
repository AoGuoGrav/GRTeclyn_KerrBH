/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef CHIGRADIENTTAGGER_HPP_
#define CHIGRADIENTTAGGER_HPP_

// Tagging criterion identical to GRChombo's ChiTaggingCriterion
// (GRChombo/Source/TaggingCriteria/ChiTaggingCriterion.hpp), which is what the
// GRChombo KerrBH example uses:
//
//      criterion = dx * |grad chi| / chi^2
//
// with 4th-order centred first derivatives. A cell is tagged for refinement
// when criterion >= regrid threshold.
//
// (GRTeclyn's own ChiTagger uses dx * |d_i d_j chi| instead, which refines a
//  different region for the same threshold.)

#include "DimensionDefinitions.hpp"
#include "FourthOrderDerivatives.hpp"
#include "CCZ4StateVariables.hpp"
#include "Tensor.hpp"

#include <AMReX_Array4.H>
#include <AMReX_Gpu.H>
#include <AMReX_REAL.H>
#include <AMReX_TagBox.H>

class ChiGradientTagger
{
  protected:
    amrex::Real m_dx;
    FourthOrderDerivatives m_deriv;
    amrex::Real m_threshold;

  public:
    // NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
    ChiGradientTagger(const amrex::Real dx, const amrex::Real a_threshold)
        : m_dx(dx), m_deriv(dx), m_threshold(a_threshold)
    {
    }

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE void
    operator()(int i, int j, int k,
               const amrex::Array4<amrex::TagBox::TagType> &tags,
               const amrex::Array4<amrex::Real const> &state) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const amrex::Real chi = state(i, j, k, c_chi);
        const auto d1_chi     = m_deriv.d1_scalar(i, j, k, state, c_chi);

        amrex::Real mod_d1_chi = 0.0;
        FOR (idir)
        {
            mod_d1_chi += d1_chi(idir) * d1_chi(idir);
        }
        const amrex::Real criterion =
            m_dx * std::sqrt(mod_d1_chi) / (chi * chi);

        if (criterion >= m_threshold)
        {
            tags(i, j, k) = amrex::TagBox::SET;
        }
    }
};

#endif /* CHIGRADIENTTAGGER_HPP_ */
