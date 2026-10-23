/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#if !defined(CHIHYPERBOLICRELAXATION_HPP_)
#error "This file should only be included through ChiHyperbolicRelaxation.hpp"
#endif

#ifndef CHIHYPERBOLICRELAXATION_IMPL_HPP_
#define CHIHYPERBOLICRELAXATION_IMPL_HPP_

template <class matter_t>
ChiHyperbolicRelaxation<matter_t>::ChiHyperbolicRelaxation(double dx,
                                                           double coefficient_A,
                                                           int a_c_theta,
                                                           double eta_damping,
                                                           double G_Newton)
    // The parent relaxation speed is one, so that it writes Ham / chi
    : ChiRelaxation<matter_t>(dx, 1.0, G_Newton),
      m_coefficient_A(coefficient_A), m_eta_damping(eta_damping),
      m_c_theta(a_c_theta)
{
}

template <class matter_t>
template <class data_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
ChiHyperbolicRelaxation<matter_t>::compute(
    int i, int j, int k, const amrex::Array4<data_t> &rhs,
    const amrex::Array4<data_t const> &state) const
{
    // Sets all RHS to zero apart from rhs_chi = Ham / chi
    ChiRelaxation<matter_t>::compute(i, j, k, rhs, state);

    const data_t ham_over_chi = rhs(i, j, k, c_chi);
    const data_t theta        = state(i, j, k, m_c_theta);

    rhs(i, j, k, c_chi) = theta;
    rhs(i, j, k, m_c_theta) =
        m_coefficient_A * ham_over_chi - m_eta_damping * theta;
}

#endif /* CHIHYPERBOLICRELAXATION_IMPL_HPP_ */
