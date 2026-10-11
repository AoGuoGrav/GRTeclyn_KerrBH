/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef EKPYROTICINITIALDATA_HPP_
#define EKPYROTICINITIALDATA_HPP_

#include "Coordinates.hpp"
#include "GRParmParse.hpp"
#include "Potential.hpp"
#include "StateVariables.hpp"
#include "Tensor.hpp"

#include <AMReX.H>
#include <AMReX_Array4.H>
#include <AMReX_REAL.H>

#include <array>
#include <cmath>
#include <string>

//! Initial data for a scalar field cosmology with inhomogeneities along x
/*!
    Follows Xue, Garfinkle, Pretorius & Steinhardt, arXiv:1308.3044, Eqs.
    (25)-(36), restricted to a single scalar field. The free data are

      Q(x)   = Pi0 + f0 cos(m x),          phi(x) = phi0 + f1 cos(m x),
      A11(x) = kappa^2 [-Pi0 f1 cos(m x) - f0 f1 cos(2 m x) / 4],
      A^Xue  = diag(A11, lambda A11, -(1 + lambda) A11),

    with kappa^2 = 8 pi G (Xue et al. use kappa^2 = 1). This choice solves
    the momentum constraint d^i A^Xue_ij = -kappa^2 Q d_j phi exactly in flat
    space. The metric is conformally flat (h_ij = delta_ij, chi = 1 before
    relaxation), K = K0 is constant, lapse = 1, shift = 0 and Gamma^i = 0.
    The stored variables are

      Pi = chi^{3/2} Q,     A_ij = chi^{3/2} A^Xue_ij,

    so the momentum constraint stays satisfied for any chi(x). The
    Hamiltonian constraint is then solved by relaxing chi (see
    EkpyroticLevel). The wavenumber is m = 2 pi n / L_x so that the data are
    periodic in x.
*/
class EkpyroticInitialData
{
  public:
    struct params_t
    {
        amrex::Real phi0{0.0};
        amrex::Real Pi0{0.0};
        bool use_scaling_Pi0{true};
        amrex::Real f0{0.0};
        amrex::Real f1{0.0};
        amrex::Real lambda{0.0};
        int mode_number{1};

        static void check_params();

        void fill_params()
        {
            GRParmParse pp("initial_data");
            pp.get("phi0", phi0);
            pp.get("Pi0", Pi0);
            pp.get("use_scaling_Pi0", use_scaling_Pi0);
            pp.get("f0", f0);
            pp.get("f1", f1);
            pp.get("lambda", lambda);
            pp.get("mode_number", mode_number);
        }
    };

    //! Quantities derived from the parameters (computed on the host)
    struct background_t
    {
        amrex::Real kappa2{};     //!< 8 pi G
        amrex::Real Pi0{};        //!< background Pi (given or scaling)
        amrex::Real wavenumber{}; //!< m = 2 pi n / L_x
        amrex::Real mean_rho{};   //!< <rho> over one period (chi = 1)
        amrex::Real K0{};         //!< sqrt(3 kappa^2 <rho>)
    };

    explicit EkpyroticInitialData(amrex::Real a_dx) : m_dx(a_dx)
    {
        m_params.fill_params();
        m_background = compute_background(m_params, m_potential);
    }

    //! Pi0 of the ekpyrotic scaling solution, or a negative number if the
    //! scaling solution does not exist for these parameters
    static amrex::Real scaling_Pi0_squared(amrex::Real a_kappa2,
                                           const Potential &a_potential,
                                           amrex::Real a_phi0)
    {
        const amrex::Real c = a_potential.get_params().c;
        return 2.0 * c * c * a_potential.V_of_phi(a_phi0) /
               (6.0 * a_kappa2 - c * c);
    }

    static background_t compute_background(const params_t &a_params,
                                           const Potential &a_potential)
    {
        background_t background;

        GRParmParse scalar_field_pp("scalar_field");
        amrex::Real G_Newton{};
        scalar_field_pp.get("G_Newton", G_Newton);
        background.kappa2 = 8.0 * M_PI * G_Newton;

        if (a_params.use_scaling_Pi0)
        {
            const amrex::Real Pi0_squared = scaling_Pi0_squared(
                background.kappa2, a_potential, a_params.phi0);
            // Pi0 < 0 for a contracting scaling solution
            background.Pi0 = -std::sqrt(std::max(Pi0_squared, 0.0));
        }
        else
        {
            background.Pi0 = a_params.Pi0;
        }

        GRParmParse geometry_pp("geometry");
        std::array<amrex::Real, AMREX_SPACEDIM> prob_extent{};
        geometry_pp.get("prob_extent", prob_extent);
        background.wavenumber =
            2.0 * M_PI * a_params.mode_number / prob_extent[0];

        // Average energy density over one period with chi = 1. The integrand
        // is smooth and periodic, so the uniform (trapezoidal) sum converges
        // spectrally.
        constexpr int num_samples = 4096;
        const amrex::Real period  = 2.0 * M_PI / background.wavenumber;
        amrex::Real rho_sum       = 0.0;
        for (int isample = 0; isample < num_samples; ++isample)
        {
            const amrex::Real x      = period * isample / num_samples;
            const amrex::Real cos_mx = std::cos(background.wavenumber * x);
            const amrex::Real sin_mx = std::sin(background.wavenumber * x);
            const amrex::Real Q      = background.Pi0 + a_params.f0 * cos_mx;
            const amrex::Real phi    = a_params.phi0 + a_params.f1 * cos_mx;
            const amrex::Real dphi_dx =
                -a_params.f1 * background.wavenumber * sin_mx;
            rho_sum += 0.5 * Q * Q + 0.5 * dphi_dx * dphi_dx +
                       a_potential.V_of_phi(phi);
        }
        background.mean_rho = rho_sum / num_samples;
        background.K0 =
            std::sqrt(3.0 * background.kappa2 *
                      std::max(background.mean_rho, 0.0)); // contracting

        return background;
    }

    //! Set all variables at a point (including ghost cells)
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    operator()(int ix, int iy, int iz,
               const amrex::Array4<amrex::Real> &state) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const amrex::CellData<amrex::Real> state_cell_data =
            state.cellData(ix, iy, iz);
        for (int icomp = 0; icomp < state_cell_data.nComp(); ++icomp)
        {
            state_cell_data[icomp] = 0.0;
        }

        state_cell_data[c_chi] = 1.0;
        FOR (i)
        {
            state_cell_data[sym_var_idx(c_h11, i, i)] = 1.0;
        }
        state_cell_data[c_K]     = m_background.K0;
        state_cell_data[c_lapse] = 1.0;
        state_cell_data[c_phi]   = phi(x_coord(ix));

        // Pi and A_ij from chi = 1
        set_momenta(ix, iy, iz, state);
    }

    //! Set Pi = chi^{3/2} Q and A_ij = chi^{3/2} A^Xue_ij from the current
    //! chi so that the momentum constraint is satisfied exactly
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_DEVICE AMREX_FORCE_INLINE void
    set_momenta(int ix, int iy, int iz,
                const amrex::Array4<amrex::Real> &state) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        const amrex::CellData<amrex::Real> state_cell_data =
            state.cellData(ix, iy, iz);
        const amrex::Real chi        = state_cell_data[c_chi];
        const amrex::Real chi_factor = chi * std::sqrt(chi);
        const amrex::Real x          = x_coord(ix);

        state_cell_data[c_Pi] = chi_factor * Q(x);

        const amrex::Real A11  = chi_factor * A11_Xue(x);
        state_cell_data[c_A11] = A11;
        state_cell_data[c_A12] = 0.0;
        state_cell_data[c_A13] = 0.0;
        state_cell_data[c_A22] = m_params.lambda * A11;
        state_cell_data[c_A23] = 0.0;
        state_cell_data[c_A33] = -(1.0 + m_params.lambda) * A11;
    }

    [[nodiscard]] const background_t &get_background() const
    {
        return m_background;
    }

    [[nodiscard]] const params_t &get_params() const { return m_params; }

  protected:
    //! Cell-centred x coordinate, measured from prob_lo = 0
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    x_coord(int ix) const
    {
        amrex::Real x{};
        Coordinates::compute_coord(x, ix, m_dx);
        return x;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    Q(amrex::Real x) const
    {
        return m_background.Pi0 +
               m_params.f0 * std::cos(m_background.wavenumber * x);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    phi(amrex::Real x) const
    {
        return m_params.phi0 +
               m_params.f1 * std::cos(m_background.wavenumber * x);
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    A11_Xue(amrex::Real x) const
    {
        const amrex::Real mx = m_background.wavenumber * x;
        return m_background.kappa2 *
               (-m_background.Pi0 * m_params.f1 * std::cos(mx) -
                0.25 * m_params.f0 * m_params.f1 * std::cos(2.0 * mx));
    }

    amrex::Real m_dx;
    params_t m_params{};
    Potential m_potential{};
    background_t m_background{};
};

inline void EkpyroticInitialData::params_t::check_params()
{
    GRParmParse pp("initial_data");

    amrex::Real phi0{0.0};
    amrex::Real Pi0{0.0};
    bool use_scaling_Pi0{true};
    amrex::Real f0{0.0};
    amrex::Real f1{0.0};
    amrex::Real lambda{0.0};
    int mode_number{1};

    pp.queryAdd("phi0", phi0);
    pp.queryAdd("Pi0", Pi0);
    pp.queryAdd("use_scaling_Pi0", use_scaling_Pi0);
    pp.queryAdd("f0", f0);
    pp.queryAdd("f1", f1);
    pp.queryAdd("lambda", lambda);
    pp.queryAdd("mode_number", mode_number);

    if (mode_number < 1)
    {
        pp.error("mode_number", "must be >= 1");
    }

    GRParmParse geometry_pp("geometry");
    std::array<int, AMREX_SPACEDIM> is_periodic{};
    geometry_pp.get("is_periodic", is_periodic);
    if (is_periodic[0] != 1)
    {
        geometry_pp.error("is_periodic",
                          "the ekpyrotic initial data require periodic "
                          "boundary conditions in the x direction");
    }

    params_t params;
    params.fill_params();
    const Potential potential;
    const background_t background = compute_background(params, potential);

    if (use_scaling_Pi0 &&
        scaling_Pi0_squared(background.kappa2, potential, phi0) < 0.0)
    {
        pp.error("use_scaling_Pi0", "the ekpyrotic scaling solution requires "
                                    "2 c^2 V(phi0) / (6 kappa^2 - c^2) >= 0");
    }

    if (background.mean_rho <= 0.0)
    {
        amrex::Abort("EkpyroticInitialData: the average energy density <rho> "
                     "= " +
                     std::to_string(background.mean_rho) +
                     " is not positive, so K0 = sqrt(3 kappa^2 <rho>) is not "
                     "defined");
    }

    amrex::Print() << "EkpyroticInitialData: kappa^2 = " << background.kappa2
                   << ", Pi0 = " << background.Pi0
                   << ", m = " << background.wavenumber
                   << ", <rho> = " << background.mean_rho
                   << ", K0 = " << background.K0 << "\n";
}

#endif /* EKPYROTICINITIALDATA_HPP_ */
