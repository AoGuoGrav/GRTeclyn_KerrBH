/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef CHIHYPERBOLICRELAXATION_HPP_
#define CHIHYPERBOLICRELAXATION_HPP_

#include "ChiRelaxation.hpp"

#include <AMReX_Array4.H>

//! Calculates the RHS for a hyperbolic (damped wave) relaxation of chi
/*!
    Like ChiRelaxation, this is used to find initial data that satisfy the
    Hamiltonian constraint, but with an auxiliary variable theta:

      rhs_chi   = theta,
      rhs_theta = A * Ham / chi - eta * theta,

    and all other RHS zero. Combined this is the damped wave equation
    d_tau^2 chi + eta d_tau chi = A Ham / chi, so an explicit scheme needs
    O(N) rather than O(N^2) steps per e-fold of the slowest mode. Ham / chi is
    computed by the parent class with its relaxation speed set to one.
    The index of theta is passed in, as it is only defined by the examples
    that use this class.
    \sa ChiRelaxation
*/
template <class matter_t>
class ChiHyperbolicRelaxation : public ChiRelaxation<matter_t>
{
  public:
    //! coefficient_A: A, eta_damping: eta, a_c_theta: the index of theta
    ChiHyperbolicRelaxation(double dx, double coefficient_A, int a_c_theta,
                            double eta_damping, double G_Newton = 1.0);

    //! Writes the RHS of all NUM_VARS components at a cell
    template <class data_t>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    compute(int i, int j, int k, const amrex::Array4<data_t> &rhs,
            const amrex::Array4<data_t const> &state) const;

  protected:
    double m_coefficient_A; //!< Coefficient A of Ham / chi in rhs_theta
    double m_eta_damping;   //!< Damping coefficient eta in rhs_theta
    int m_c_theta;          //!< Index of the auxiliary variable theta
};

#include "ChiHyperbolicRelaxation.impl.hpp"

#endif /* CHIHYPERBOLICRELAXATION_HPP_ */
