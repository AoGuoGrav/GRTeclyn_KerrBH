/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef ANISOTROPYDIAGNOSTIC_HPP_
#define ANISOTROPYDIAGNOSTIC_HPP_

#include "CCZ4Geometry.hpp"
#include "FourthOrderDerivatives.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"
#include "StateVariables.hpp"
#include "TensorAlgebra.hpp"

#include <AMReX_AmrLevel.H>
#include <AMReX_Array4.H>
#include <AMReX_MultiFab.H>

#include <string>

//! Derived variable with the shear and energy density of the scalar field
/*!
    sigma2 = (1/2) sigma_ij sigma^ij, where sigma_ij is the trace-free part of
    the physical extrinsic curvature. In terms of the BSSN variables this is
    (1/2) A_ij A^ij with indices raised by the conformal metric h^ij. rho is
    the energy density of the scalar field (without the factor 8 pi G) and the
    third component is sigma2 / rho.
*/
class AnisotropyDiagnostic
{
  public:
    using matter_t = ScalarField<Potential, FourthOrderDerivatives>;

    /// derive record name
    static inline const std::string name = "anisotropy";

    /// Variable names
    static inline const amrex::Vector<std::string> var_names = {
        "sigma2", "rho", "sigma2_over_rho"};

    explicit AnisotropyDiagnostic(amrex::Real a_dx) : m_deriv(a_dx) {}

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(int ix, int iy, int iz, const amrex::Array4<amrex::Real> &out,
               const amrex::Array4<const amrex::Real> &state, int dcomp) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        // ScalarFieldVars keeps a reference to the cell data, so it must
        // outlive vars (do not pass a temporary)
        const amrex::CellData<const amrex::Real> state_cell_data =
            state.cellData(ix, iy, iz);
        const ScalarFieldVars vars(state_cell_data);
        const Tensor::Rank2 h_UU = CCZ4Geometry::compute_inverse_metric(vars);

        Tensor::Rank2 A_LL{};
        FOR (i, j)
        {
            A_LL(i, j) = vars.A(i, j);
        }
        const Tensor::Rank2 A_UU = TensorAlgebra::raise_all(A_LL, h_UU);
        const amrex::Real sigma2 =
            0.5 * TensorAlgebra::compute_trace(A_LL, A_UU);

        const amrex::Real rho =
            m_matter.compute_emtensor(ix, iy, iz, state, m_deriv, h_UU).rho;

        out(ix, iy, iz, dcomp)     = sigma2;
        out(ix, iy, iz, dcomp + 1) = rho;
        out(ix, iy, iz, dcomp + 2) = (rho != 0.0) ? sigma2 / rho : 0.0;
    }

    static void set_up(int a_state_index)
    {
        const int num_ghosts = 2;
        auto &derive_lst     = amrex::AmrLevel::get_derive_lst();
        const auto &desc_lst = amrex::AmrLevel::get_desc_lst();

        derive_lst.add(
            name, amrex::IndexType::TheCellType(),
            static_cast<int>(var_names.size()), var_names,
            AnisotropyDiagnostic::compute_mf,
            [=](const amrex::Box &box) { return amrex::grow(box, num_ghosts); },
            &amrex::cell_quartic_interp);
        derive_lst.addComponent(name, desc_lst, a_state_index, 0, NUM_VARS);
    }

    // Has signature of DeriveFuncMF so that it can be stored in the derive_lst
    static void compute_mf(amrex::MultiFab &out_mf, int dcomp, int ncomp,
                           const amrex::MultiFab &state_mf,
                           const amrex::Geometry &geomdata,
                           amrex::Real /*time*/, const int * /*bcrec*/,
                           int /*level*/)
    {
        AMREX_ALWAYS_ASSERT(ncomp == static_cast<int>(var_names.size()));

        const auto &out_arrays   = out_mf.arrays();
        const auto &state_arrays = state_mf.const_arrays();
        const AnisotropyDiagnostic anisotropy(geomdata.CellSize(0));

        amrex::ParallelFor(
            out_mf, out_mf.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz) noexcept
            {
                anisotropy(ix, iy, iz, out_arrays[box_no], state_arrays[box_no],
                           dcomp);
            });
    }

  protected:
    matter_t m_matter;
    FourthOrderDerivatives m_deriv;
};

#endif /* ANISOTROPYDIAGNOSTIC_HPP_ */
