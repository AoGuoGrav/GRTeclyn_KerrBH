/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef XCTSRELAXATION_HPP_
#define XCTSRELAXATION_HPP_

#include "DimensionDefinitions.hpp"
#include "EkpyroticBHData.hpp"
#include "FourthOrderDerivatives.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"

#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

//! Hyperbolic relaxation of the conformally flat XCTS equations
/*!
    Unknowns: psi = 1 + f + u, alpha psi = 1 + f + v (f is the analytic
    puncture part, see EkpyroticBHData) and the shift beta^i. The traceless
    part is

      Ahat_ij = Ahat_free_ij + w (Lbeta)_ij,
      (Lbeta)_ij = d_i beta_j + d_j beta_i - 2/3 delta_ij d_k beta^k,

    with the weight w = 1 / (2 alpha_tilde_reg) = (1 + u)^7 / (2 (1 + v)),
    i.e. XCTS with alpha_tilde = psi^-6 alpha evaluated on the regular parts
    of psi and alpha psi, and u_ij = d_t gamma_tilde_ij = -2 alpha_tilde
    Ahat_free_ij. This is exactly XCTS for M = 0 and wherever f = 0
    (r > bh.window_outer). Closer to the puncture it is still an exact
    solution of the constraints (a weighted CTS choice of free data) but
    avoids the weight psi^6 / alpha ~ r^-6: with it the preconditioned
    momentum operator acquires a first-order term ~ 7 / r, is no longer
    self-adjoint and the hyperbolic relaxation of beta becomes unstable.

    K = K0 is constant, so the residuals of the three equations are

      R_u    = Lap(psi) - [K0^2 psi^5 / 12 - psi^-7 Ahat.Ahat / 8
                           - kappa^2 psi^5 rho / 4],
      R_v    = Lap(alpha psi) - alpha psi [7/8 psi^-8 Ahat.Ahat
                + 5/12 psi^4 K0^2 + kappa^2 psi^4 (rho + 2 S) / 4]
                + psi^5 dtK,
      R_beta = d_j Ahat_ij - kappa^2 psi^6 S_i.

    Each unknown X obeys d_tau X = theta_X, d_tau theta_X = A_X P_X R_X -
    eta_X theta_X with the preconditioners P_u = 1, P_v = psi^-4 (removes
    the psi^4 mass term that diverges at the puncture) and P_beta = 1 / w.
    With solve_lapse = false the lapse is not solved for:
    v follows u exactly so alpha = 1 (this is CTS with alpha_tilde =
    psi^-6).
*/
class XCTSRelaxation
{
  public:
    struct coefficients_t
    {
        amrex::Real A_u{};
        amrex::Real A_v{};
        amrex::Real A_beta{};
        amrex::Real eta_u{};
        amrex::Real eta_v{};
        amrex::Real eta_beta{};
        bool solve_lapse{true};
    };

    //! Everything needed for the residuals and the BSSN variables at a point
    struct point_t
    {
        Tensor::Rank1 displacement{};
        amrex::Real psi{};
        amrex::Real alpha_psi{};
        amrex::Real lapse{};
        Tensor::Rank1 beta{};
        amrex::Real shift_weight{}; //!< w = (1 + u)^7 / (2 (1 + v))
        Tensor::Rank2 Ahat{};
        amrex::Real Ahat_squared{};
        Tensor::Rank2 L_beta{};
        Tensor::Rank1 Ahat_free_divergence{};
        EkpyroticBHData::matter_t matter{};
        amrex::Real rho{};
        amrex::Real trace_S{};
    };

    struct residuals_t
    {
        amrex::Real u{};
        amrex::Real v{};
        Tensor::Rank1 beta{};
    };

    XCTSRelaxation(const EkpyroticBHData &a_data, amrex::Real a_dx,
                   const coefficients_t &a_coefficients)
        : m_data(a_data), m_deriv(a_dx), m_coefficients(a_coefficients)
    {
    }

    //! Write the relaxation RHS (all other RHS components must be zero)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    compute(int ix, int iy, int iz, const amrex::Array4<amrex::Real> &rhs,
            const amrex::Array4<const amrex::Real> &state) const
    {
        point_t point;
        const residuals_t res = residuals(ix, iy, iz, state, point);

        const amrex::Real psi2 = point.psi * point.psi;

        rhs(ix, iy, iz, c_xcts_u) = state(ix, iy, iz, c_theta_u);
        rhs(ix, iy, iz, c_theta_u) =
            m_coefficients.A_u * res.u -
            m_coefficients.eta_u * state(ix, iy, iz, c_theta_u);

        if (m_coefficients.solve_lapse)
        {
            rhs(ix, iy, iz, c_xcts_v) = state(ix, iy, iz, c_theta_v);
            rhs(ix, iy, iz, c_theta_v) =
                m_coefficients.A_v * res.v / (psi2 * psi2) -
                m_coefficients.eta_v * state(ix, iy, iz, c_theta_v);
        }
        else
        {
            // v = u (alpha = 1) for all pseudo-times
            rhs(ix, iy, iz, c_xcts_v)  = rhs(ix, iy, iz, c_xcts_u);
            rhs(ix, iy, iz, c_theta_v) = rhs(ix, iy, iz, c_theta_u);
        }

        FOR (i)
        {
            rhs(ix, iy, iz, c_shift1 + i) =
                state(ix, iy, iz, c_theta_beta1 + i);
            rhs(ix, iy, iz, c_theta_beta1 + i) =
                m_coefficients.A_beta * res.beta(i) / point.shift_weight -
                m_coefficients.eta_beta * state(ix, iy, iz, c_theta_beta1 + i);
        }
    }

    //! Residuals of the three XCTS equations (and the point quantities)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE residuals_t residuals(
        int ix, int iy, int iz, const amrex::Array4<const amrex::Real> &state,
        point_t &a_point) const
    {
        const auto d1_u    = m_deriv.d1_scalar(ix, iy, iz, state, c_xcts_u);
        const auto d1_v    = m_deriv.d1_scalar(ix, iy, iz, state, c_xcts_v);
        const auto d2_u    = m_deriv.d2_scalar(ix, iy, iz, state, c_xcts_u);
        const auto d2_v    = m_deriv.d2_scalar(ix, iy, iz, state, c_xcts_v);
        const auto d1_beta = m_deriv.d1_vector(ix, iy, iz, state, c_shift1);
        const auto d2_beta = m_deriv.d2_vector(ix, iy, iz, state, c_shift1);

        const auto puncture = fill_point(ix, iy, iz, state, d1_beta, a_point);
        const amrex::Real kappa2 = m_data.get_derived().kappa2;
        const amrex::Real K0     = m_data.get_derived().K0;

        const amrex::Real psi  = a_point.psi;
        const amrex::Real psi2 = psi * psi;
        const amrex::Real psi4 = psi2 * psi2;
        const amrex::Real psi5 = psi4 * psi;
        const amrex::Real psi6 = psi4 * psi2;
        const amrex::Real psi7 = psi6 * psi;

        amrex::Real lap_u = 0.0;
        amrex::Real lap_v = 0.0;
        FOR (i)
        {
            lap_u += d2_u(i, i);
            lap_v += d2_v(i, i);
        }

        residuals_t res;

        // Hamiltonian constraint
        res.u = lap_u + puncture.laplacian_f -
                (K0 * K0 * psi5 / 12.0 - a_point.Ahat_squared / (8.0 * psi7) -
                 0.25 * kappa2 * psi5 * a_point.rho);

        // Lapse equation (d_t K = dtK, d_i K = 0)
        res.v =
            lap_v + puncture.laplacian_f -
            a_point.alpha_psi *
                (0.875 * a_point.Ahat_squared / (psi4 * psi4) +
                 5.0 / 12.0 * psi4 * K0 * K0 +
                 0.25 * kappa2 * psi4 * (a_point.rho + 2.0 * a_point.trace_S)) +
            psi5 * m_data.get_derived().dtK;

        // Momentum constraint: d_j Ahat_ij = d_j Ahat_free_ij
        //   + w [Lap beta_i + 1/3 d_i d_k beta^k] + (Lbeta)_ij d_j w
        // with w = (1 + u)^7 / (2 (1 + v))
        const amrex::Real weight        = a_point.shift_weight;
        const amrex::Real psi_reg       = 1.0 + state(ix, iy, iz, c_xcts_u);
        const amrex::Real alpha_psi_reg = 1.0 + state(ix, iy, iz, c_xcts_v);
        Tensor::Rank1 d1_weight{};
        FOR (j)
        {
            d1_weight(j) =
                weight * (7.0 * d1_u(j) / psi_reg - d1_v(j) / alpha_psi_reg);
        }
        FOR (i)
        {
            amrex::Real lap_beta    = 0.0;
            amrex::Real grad_div    = 0.0;
            amrex::Real L_dot_dcoef = 0.0;
            FOR (j)
            {
                lap_beta    += d2_beta(i, j, j);
                grad_div    += d2_beta(j, i, j);
                L_dot_dcoef += a_point.L_beta(i, j) * d1_weight(j);
            }
            const amrex::Real S_i =
                -a_point.matter.Pi * a_point.matter.d1_phi(i);
            res.beta(i) = a_point.Ahat_free_divergence(i) +
                          weight * (lap_beta + grad_div / 3.0) + L_dot_dcoef -
                          kappa2 * psi6 * S_i;
        }

        return res;
    }

    //! Fill the point quantities (psi, lapse, Ahat, matter) given d_j beta^i
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE EkpyroticBHData::puncture_t
    fill_point(int ix, int iy, int iz,
               const amrex::Array4<const amrex::Real> &state,
               const Tensor::Rank2 &a_d1_beta, point_t &a_point) const
    {
        a_point.displacement = m_data.displacement(ix, iy, iz);
        const auto puncture  = m_data.puncture(a_point.displacement);

        const amrex::Real u = state(ix, iy, iz, c_xcts_u);
        const amrex::Real v = state(ix, iy, iz, c_xcts_v);
        a_point.psi         = 1.0 + puncture.f + u;
        a_point.alpha_psi   = 1.0 + puncture.f + v;
        a_point.lapse       = a_point.alpha_psi / a_point.psi;
        FOR (i)
        {
            a_point.beta(i) = state(ix, iy, iz, c_shift1 + i);
        }

        const amrex::Real psi2     = a_point.psi * a_point.psi;
        const amrex::Real psi4     = psi2 * psi2;
        const amrex::Real psi_reg  = 1.0 + u;
        const amrex::Real psi_reg2 = psi_reg * psi_reg;
        a_point.shift_weight =
            psi_reg2 * psi_reg2 * psi_reg2 * psi_reg / (2.0 * (1.0 + v));

        Tensor::Rank1 Ahat_diagonal{};
        m_data.Ahat_free(ix, a_point.displacement, Ahat_diagonal,
                         a_point.Ahat_free_divergence);
        amrex::Real div_beta = 0.0;
        FOR (k)
        {
            div_beta += a_d1_beta(k, k);
        }
        a_point.Ahat_squared = 0.0;
        FOR (i, j)
        {
            // d1_beta(i, j) = d_j beta^i
            a_point.L_beta(i, j) = a_d1_beta(i, j) + a_d1_beta(j, i) -
                                   ((i == j) ? 2.0 / 3.0 * div_beta : 0.0);
            a_point.Ahat(i, j)   = a_point.shift_weight * a_point.L_beta(i, j) +
                                   ((i == j) ? Ahat_diagonal(i) : 0.0);
            a_point.Ahat_squared += a_point.Ahat(i, j) * a_point.Ahat(i, j);
        }

        a_point.matter = m_data.matter(ix, a_point.displacement, 1.0 + u);
        amrex::Real grad_phi_squared = 0.0;
        FOR (i)
        {
            grad_phi_squared +=
                a_point.matter.d1_phi(i) * a_point.matter.d1_phi(i);
        }
        const amrex::Real Pi2 = a_point.matter.Pi * a_point.matter.Pi;
        a_point.rho =
            0.5 * Pi2 + 0.5 * grad_phi_squared / psi4 + a_point.matter.V;
        a_point.trace_S =
            1.5 * Pi2 - 0.5 * grad_phi_squared / psi4 - 3.0 * a_point.matter.V;
        return puncture;
    }

    //! BSSN variables from the XCTS unknowns, written into a_out at a point
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    set_bssn_variables(int ix, int iy, int iz,
                       const amrex::Array4<amrex::Real> &a_out,
                       const amrex::Array4<const amrex::Real> &a_in) const
    {
        const auto d1_beta = m_deriv.d1_vector(ix, iy, iz, a_in, c_shift1);
        point_t point;
        fill_point(ix, iy, iz, a_in, d1_beta, point);
        write_bssn_variables(ix, iy, iz, a_out, point);
    }

    //! As set_bssn_variables, for beta = 0 (no derivatives needed)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    set_bssn_variables_zero_shift(int ix, int iy, int iz,
                                  const amrex::Array4<amrex::Real> &a_out) const
    {
        const Tensor::Rank2 zero_d1_beta{};
        point_t point;
        fill_point(ix, iy, iz, a_out, zero_d1_beta, point);
        write_bssn_variables(ix, iy, iz, a_out, point);
    }

  protected:
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    write_bssn_variables(int ix, int iy, int iz,
                         const amrex::Array4<amrex::Real> &a_out,
                         const point_t &a_point) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const amrex::Real psi2 = a_point.psi * a_point.psi;
        const amrex::Real psi6 = psi2 * psi2 * psi2;

        a_out(ix, iy, iz, c_chi) = 1.0 / (psi2 * psi2);
        FOR (i, j)
        {
            a_out(ix, iy, iz, sym_var_idx(c_h11, i, j)) = (i == j) ? 1.0 : 0.0;
            a_out(ix, iy, iz, sym_var_idx(c_A11, i, j)) =
                a_point.Ahat(i, j) / psi6;
        }
        a_out(ix, iy, iz, c_K)     = m_data.get_derived().K0;
        a_out(ix, iy, iz, c_Theta) = 0.0;
        FOR (i)
        {
            a_out(ix, iy, iz, c_Gamma1 + i) = 0.0;
            a_out(ix, iy, iz, c_B1 + i)     = 0.0;
        }
        a_out(ix, iy, iz, c_lapse) = a_point.lapse;
        a_out(ix, iy, iz, c_phi)   = a_point.matter.phi;
        a_out(ix, iy, iz, c_Pi)    = a_point.matter.Pi;
    }

    EkpyroticBHData m_data;
    FourthOrderDerivatives m_deriv;
    coefficients_t m_coefficients;
};

#endif /* XCTSRELAXATION_HPP_ */
