/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#if !defined(CHIRELAXATION_HPP_)
#error "This file should only be included through ChiRelaxation.hpp"
#endif

#ifndef CHIRELAXATION_IMPL_HPP_
#define CHIRELAXATION_IMPL_HPP_

template <class matter_t>
ChiRelaxation<matter_t>::ChiRelaxation(double dx, double relax_speed,
                                       double G_Newton)
    : m_relax_speed(relax_speed), m_G_Newton(G_Newton), m_deriv(dx)
{
}

template <class matter_t>
template <class data_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE void ChiRelaxation<matter_t>::compute(
    int i, int j, int k, const amrex::Array4<data_t> &rhs_arrays,
    const amrex::Array4<data_t const> &state_arrays) const
{

    // copy data from the gridpoint into local variable and calculate derivs
    const Vars vars(state_arrays.cellData(i, j, k));
    const auto d1_chi   = m_deriv.d1_scalar(i, j, k, state_arrays, c_chi);
    const auto d1_Gamma = m_deriv.d1_vector(i, j, k, state_arrays, c_Gamma1);
    const auto d1_h     = m_deriv.d1_sym_tensor(i, j, k, state_arrays, c_h11);
    const auto d2_chi   = m_deriv.d2_scalar(i, j, k, state_arrays, c_chi);
    const auto d2_h     = m_deriv.d2_sym_tensor(i, j, k, state_arrays, c_h11);

    // All components that are not explicitly set in rhs_equation are 0
    const amrex::CellData<data_t> rhs_cell_data = rhs_arrays.cellData(i, j, k);
    for (int icomp = 0; icomp < rhs_cell_data.nComp(); ++icomp)
    {
        rhs_cell_data[icomp] = 0.;
    }

    // Write the rhs into the output FArrayBox
    rhs_cell_data[c_chi] = rhs_equation(i, j, k, state_arrays, vars, d1_chi,
                                        d1_Gamma, d1_h, d2_chi, d2_h);
}

template <class matter_t>
AMREX_GPU_DEVICE AMREX_FORCE_INLINE amrex::Real
ChiRelaxation<matter_t>::rhs_equation(
    int i, int j, int k, const amrex::Array4<amrex::Real const> &state,
    const Vars &vars, const Tensor::Rank1 &d1_chi,
    const Tensor::Rank2 &d1_Gamma, const Tensor::Sym12Rank3 &d1_h,
    const Tensor::Sym12Rank2 &d2_chi, const Tensor::Sym12Sym34Rank4 &d2_h) const
{

    const auto h_UU  = CCZ4Geometry::compute_inverse_metric(vars);
    const auto chris = CCZ4Geometry::compute_christoffel(d1_h, h_UU);

    // Calculate elements of the decomposed stress energy tensor and ricci
    // tensor
    const auto emtensor =
        my_matter.compute_emtensor(i, j, k, state, m_deriv, h_UU);
    const auto ricci = CCZ4Geometry::compute_ricci(vars, d1_chi, d1_Gamma, d1_h,
                                                   d2_chi, d2_h, h_UU, chris);
    const auto A_UU  = CCZ4Geometry::compute_A_UU(vars, h_UU);
    const amrex::Real tr_AA =
        CCZ4Geometry::compute_Aij_squared_with_A_UU(vars, A_UU);

    // Calculate the relaxation RHS for chi, all other vars RHS zero
    // Could have called ConstraintsMatter here, but it is hardly worth it
    // The division by chi prevents it (usually) from doing chi=0 as a solution
    return m_relax_speed *
           (ricci.scalar +
            (GR_SPACEDIM - 1.) * vars.K() * vars.K() / GR_SPACEDIM - tr_AA -
            16.0 * M_PI * m_G_Newton * emtensor.rho) /
           vars.chi();
}

#endif /* CHIRELAXATION_IMPL_HPP_ */
