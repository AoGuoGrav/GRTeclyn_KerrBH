/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef CHIRELAXATION_HPP_
#define CHIRELAXATION_HPP_

#include "CCZ4Geometry.hpp"
#include "FourthOrderDerivatives.hpp"
#include "StateVariables.hpp" //This files needs NUM_VARS - total number of components
#include "Tensor.hpp"
#include "TensorAlgebra.hpp"

//!  Calculates RHS for relaxation of the conformal factor, for initial
//!  conditions
/*!
     The class calculates the RHS evolution for the relaxation of the conformal
     factor chi, in order to satisfy the Hamiltonian constraint for some general
     matter configuration. It assumes that the other variables (K, h_ij etc) are
     fixed and satisfy the Momentum constraint. It calculates the RHS at each
   step as the relaxation speed multiplied by the error in the Hamiltonian
   constraint. It is extremely inefficient and takes a long time to converge,
   but it works provided that there is indeed a sensible solution (note that it
   can converge to \chi = 0 everywhere if there is not, so one must always check
   it is converging on something sensible). Note that the relaxation speed
   should be set to a value less than 2/15*dx_min/courant_factor for numerical
   stability. \sa m_relax_speed()
*/

template <class matter_t> class ChiRelaxation
{

    // Use the variable definitions in the matter class
    using Vars = typename matter_t::Vars;

  public:
    //! Constructor of class ChiRelaxation
    /*!
        Takes in the grid spacing, plus the relaxation speed, a matter object
        and the value of Newton's constant, which is set to one by default.
    */
    ChiRelaxation(double dx, double relax_speed, double G_Newton = 1.0);

    //! The compute member which calculates the RHS at each point in the box \sa
    //! rhs_equation()
    template <class data_t>
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    compute(int i, int j, int k, const amrex::Array4<data_t> &rhs,
            const amrex::Array4<data_t const> &state) const;

  protected:
    matter_t my_matter;   //!< The matter object, e.g. a scalar field.
    double m_relax_speed; //!< The coefficient of the Hamiltonian used to
                          //! set relaxation speed.
    double m_G_Newton;    //!< Newton's constant, set to one by default.
    FourthOrderDerivatives
        m_deriv; //!< An object for calculating derivatives of the variables

    //! The function which calculates the RHS of chi, given the vars and
    //! derivatives \sa compute()
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real rhs_equation(
        int i, int j, int k, //!< the grid indices (for the matter EM tensor).
        const amrex::Array4<amrex::Real const>
            &state,       //!< the state data (for the matter EM tensor).
        const Vars &vars, //!< the value of the variables at the point.
        const Tensor::Rank1 &d1_chi, //!< first derivatives of chi.
        const Tensor::Rank2 &d1_Gamma, //!< first derivatives of Gamma^i.
        const Tensor::Sym12Rank3 &d1_h, //!< first derivatives of h_ij.
        const Tensor::Sym12Rank2 &d2_chi, //!< second derivatives of chi.
        const Tensor::Sym12Sym34Rank4 &d2_h //!< second derivatives of h_ij.
    ) const;
};

#include "ChiRelaxation.impl.hpp"

#endif /* CHIRELAXATION_HPP_ */
