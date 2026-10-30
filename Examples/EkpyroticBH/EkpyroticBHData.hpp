/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef EKPYROTICBHDATA_HPP_
#define EKPYROTICBHDATA_HPP_

#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"
#include "GRParmParse.hpp"
#include "Potential.hpp"
#include "Tensor.hpp"

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

//! Free data for the XCTS initial data of a black hole in an ekpyrotic
//! (scalar field) cosmology, periodic in all directions
/*!
    Conventions: kappa^2 = 8 pi G, gamma_ij = psi^4 delta_ij, physical
    A^ij = psi^-10 Ahat^ij (so the BSSN variable is A_ij = psi^-6 Ahat_ij),
    rho = Pi^2/2 + psi^-4 |d phi|^2/2 + V, S_i = -Pi d_i phi.

    The black hole is a puncture of mass parameter M at bh.center:

      psi = 1 + f + u,   f = M W(r) / (2 r),

    where W is a smooth window (W = 1 for r < bh.window_inner, W = 0 for
    r > bh.window_outer, so f is periodic) and u is regular and solved for.

    Matter (fixed physical functions, Xue et al. arXiv:1308.3044 far from the
    black hole):

      phi = phi0 + f1 B(r) cos(m x),
      Pi  = B psi_reg^-6 (Pi0 + f0 B cos(m x)) + (1 - B) Pi_BH,

    with psi_reg = 1 + u and B a second window (B = 0 for
    r < bh.homogeneous_inner, B = 1 for r > bh.homogeneous_outer). The
    scaling with psi_reg (not psi) keeps Pi finite at the puncture while
    reproducing exactly the York scaling of EkpyroticInitialData when M = 0
    (then B = 1 and f = 0).

    The trace K = K0 is constant. In the core r < bh.homogeneous_inner the
    matter is homogeneous with 3 kappa^2 rho_BH = K0^2 (local Friedmann), so
    the psi^5 terms of the Hamiltonian constraint cancel identically there
    and psi = 1 + M / (2 r) is a local solution. Periodicity of the
    Hamiltonian constraint (int Laplacian(psi) = 0, while the window region
    gives int Laplacian(f) = 2 pi M) is satisfied by including the black hole
    mass in the mean density:

      K0^2 = 3 kappa^2 <rho>_{psi = 1} + 24 pi M / V_box.

    The remaining small mismatch is absorbed by the mean of u.

    The free traceless part is the Xue et al. solution of the flat momentum
    constraint, damped by B: Ahat_free = B diag(A11, lambda A11,
    -(1 + lambda) A11). The XCTS shift absorbs the rest (see XCTSRelaxation).

    dtK, the free data of the lapse equation, is the uniform value
    K0^2 / 3 + kappa^2 (rho_BH + S_BH) / 2, for which alpha = 1 solves the
    lapse equation in the homogeneous core.
*/
class EkpyroticBHData
{
  public:
    //! Black hole parameters (read from "bh.*")
    struct bh_params_t
    {
        amrex::Real mass{0.0};
        std::array<amrex::Real, AMREX_SPACEDIM> center{};
        amrex::Real window_inner{0.0};
        amrex::Real window_outer{0.0};
        amrex::Real homogeneous_inner{0.0};
        amrex::Real homogeneous_outer{0.0};

        static void check_params();

        void fill_params()
        {
            GRParmParse pp("bh");
            pp.get("mass", mass);
            pp.get("center", center);
            pp.get("window_inner", window_inner);
            pp.get("window_outer", window_outer);
            pp.get("homogeneous_inner", homogeneous_inner);
            pp.get("homogeneous_outer", homogeneous_outer);
        }
    };

    //! Cosmological free data (read from "initial_data.*", as in the
    //! Ekpyrotic example)
    struct cosmo_params_t
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
    struct derived_t
    {
        amrex::Real kappa2{};     //!< 8 pi G
        amrex::Real Pi0{};        //!< background Pi (given or scaling)
        amrex::Real wavenumber{}; //!< m = 2 pi n / L_x
        amrex::Real mean_rho{};   //!< <rho> of the Xue data with psi = 1
        amrex::Real box_volume{};
        amrex::Real K0{};     //!< sqrt(3 kappa^2 <rho> + 24 pi M / V_box)
        amrex::Real rho_BH{}; //!< K0^2 / (3 kappa^2)
        amrex::Real Pi_BH{};  //!< Pi in the homogeneous core
        amrex::Real S_BH{};   //!< trace of S_ij in the homogeneous core
        amrex::Real dtK{};    //!< uniform d_t K of the lapse equation
        std::array<amrex::Real, AMREX_SPACEDIM> domain_lengths{};
    };

    explicit EkpyroticBHData(amrex::Real a_dx) : m_dx(a_dx)
    {
        m_bh.fill_params();
        m_cosmo.fill_params();
        m_derived = compute_derived(m_bh, m_cosmo, m_potential);
    }

    static amrex::Real scaling_Pi0_squared(amrex::Real a_kappa2,
                                           const Potential &a_potential,
                                           amrex::Real a_phi0)
    {
        const amrex::Real c_value = a_potential.get_params().c;
        return 2.0 * c_value * c_value * a_potential.V_of_phi(a_phi0) /
               (6.0 * a_kappa2 - c_value * c_value);
    }

    static derived_t compute_derived(const bh_params_t &a_bh,
                                     const cosmo_params_t &a_cosmo,
                                     const Potential &a_potential)
    {
        derived_t derived;

        GRParmParse scalar_field_pp("scalar_field");
        amrex::Real G_Newton{};
        scalar_field_pp.get("G_Newton", G_Newton);
        derived.kappa2 = 8.0 * M_PI * G_Newton;

        if (a_cosmo.use_scaling_Pi0)
        {
            const amrex::Real Pi0_squared =
                scaling_Pi0_squared(derived.kappa2, a_potential, a_cosmo.phi0);
            // Pi0 < 0 for a contracting scaling solution
            derived.Pi0 = -std::sqrt(std::max(Pi0_squared, 0.0));
        }
        else
        {
            derived.Pi0 = a_cosmo.Pi0;
        }

        GRParmParse geometry_pp("geometry");
        geometry_pp.get("prob_extent", derived.domain_lengths);
        derived.wavenumber =
            2.0 * M_PI * a_cosmo.mode_number / derived.domain_lengths[0];
        derived.box_volume = 1.0;
        for (const amrex::Real length : derived.domain_lengths)
        {
            derived.box_volume *= length;
        }

        // Average energy density of the Xue data over one period with
        // psi = 1 (uniform sum, spectrally accurate for periodic data)
        constexpr int num_samples = 4096;
        const amrex::Real period  = 2.0 * M_PI / derived.wavenumber;
        amrex::Real rho_sum       = 0.0;
        for (int isample = 0; isample < num_samples; ++isample)
        {
            const amrex::Real x     = period * isample / num_samples;
            const amrex::Real phase = derived.wavenumber * x;
            const amrex::Real Q_x = derived.Pi0 + a_cosmo.f0 * std::cos(phase);
            const amrex::Real phi = a_cosmo.phi0 + a_cosmo.f1 * std::cos(phase);
            const amrex::Real dphi_dx =
                -a_cosmo.f1 * derived.wavenumber * std::sin(phase);
            rho_sum += 0.5 * Q_x * Q_x + 0.5 * dphi_dx * dphi_dx +
                       a_potential.V_of_phi(phi);
        }
        derived.mean_rho = rho_sum / num_samples;

        // The black hole mass enters the mean density (see class comment)
        const amrex::Real K0_squared =
            3.0 * derived.kappa2 * derived.mean_rho +
            24.0 * M_PI * a_bh.mass / derived.box_volume;
        derived.K0 = std::sqrt(std::max(K0_squared, 0.0)); // contracting

        // Local Friedmann core: 3 kappa^2 rho_BH = K0^2 with phi = phi0
        const amrex::Real V0_core = a_potential.V_of_phi(a_cosmo.phi0);
        derived.rho_BH            = K0_squared / (3.0 * derived.kappa2);
        derived.Pi_BH =
            -std::sqrt(std::max(2.0 * (derived.rho_BH - V0_core), 0.0));
        derived.S_BH = 1.5 * derived.Pi_BH * derived.Pi_BH - 3.0 * V0_core;
        derived.dtK  = K0_squared / 3.0 +
                       0.5 * derived.kappa2 * (derived.rho_BH + derived.S_BH);

        return derived;
    }

    //! Displacement from the black hole (nearest periodic image)
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE Tensor::Rank1
    displacement(int ix, int iy, int iz) const
    {
        const std::array<int, 3> cell{ix, iy, iz};
        Tensor::Rank1 out{};
        FOR (idir)
        {
            amrex::Real coord{};
            Coordinates::compute_coord(coord, cell[idir], m_dx,
                                       m_bh.center[idir]);
            const amrex::Real length  = m_derived.domain_lengths[idir];
            coord                    -= length * std::round(coord / length);
            out(idir)                 = coord;
        }
        return out;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static amrex::Real
    radius(const Tensor::Rank1 &a_displacement)
    {
        amrex::Real r2 = 0.0;
        FOR (idir)
        {
            r2 += a_displacement(idir) * a_displacement(idir);
        }
        return std::max(std::sqrt(r2), 1.0e-12);
    }

    //! Puncture part f = M W / (2 r) of psi and its derivatives
    struct puncture_t
    {
        amrex::Real f{0.0};
        Tensor::Rank1 d1_f{};
        amrex::Real laplacian_f{0.0};
    };

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE puncture_t
    puncture(const Tensor::Rank1 &a_displacement) const
    {
        puncture_t out;
        if (m_bh.mass == 0.0)
        {
            return out;
        }
        const amrex::Real r = radius(a_displacement);
        amrex::Real window{};
        amrex::Real d_window{};
        amrex::Real d2_window{};
        smooth_step(r, m_bh.window_inner, m_bh.window_outer, window, d_window,
                    d2_window);
        // W = 1 - step
        window    = 1.0 - window;
        d_window  = -d_window;
        d2_window = -d2_window;

        const amrex::Real half_mass = 0.5 * m_bh.mass;
        out.f                       = half_mass * window / r;
        const amrex::Real df_dr = half_mass * (d_window / r - window / (r * r));
        FOR (idir)
        {
            out.d1_f(idir) = df_dr * a_displacement(idir) / r;
        }
        // Laplacian(W / r) = W'' / r
        out.laplacian_f = half_mass * d2_window / r;
        return out;
    }

    //! Homogenisation window B (0 near the black hole, 1 far away)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    homogenisation(const Tensor::Rank1 &a_displacement, amrex::Real &a_B,
                   Tensor::Rank1 &a_d1_B) const
    {
        FOR (idir)
        {
            a_d1_B(idir) = 0.0;
        }
        if (m_bh.mass == 0.0)
        {
            a_B = 1.0;
            return;
        }
        const amrex::Real r = radius(a_displacement);
        amrex::Real d_B{};
        amrex::Real d2_B{};
        smooth_step(r, m_bh.homogeneous_inner, m_bh.homogeneous_outer, a_B, d_B,
                    d2_B);
        FOR (idir)
        {
            a_d1_B(idir) = d_B * a_displacement(idir) / r;
        }
    }

    //! Matter at a point, given psi_reg = 1 + u (for Pi)
    struct matter_t
    {
        amrex::Real phi{};
        amrex::Real Pi{};
        Tensor::Rank1 d1_phi{};
        amrex::Real V{};
    };

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE matter_t
    matter(int ix, const Tensor::Rank1 &a_displacement,
           amrex::Real a_psi_reg) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        amrex::Real B{};
        Tensor::Rank1 d1_B{};
        homogenisation(a_displacement, B, d1_B);

        const amrex::Real phase  = m_derived.wavenumber * x_coord(ix);
        const amrex::Real cos_mx = std::cos(phase);
        const amrex::Real sin_mx = std::sin(phase);

        matter_t out;
        out.phi = m_cosmo.phi0 + m_cosmo.f1 * B * cos_mx;
        FOR (idir)
        {
            out.d1_phi(idir) = m_cosmo.f1 * d1_B(idir) * cos_mx;
        }
        out.d1_phi(0) -= m_cosmo.f1 * B * m_derived.wavenumber * sin_mx;

        const amrex::Real Q_x      = m_derived.Pi0 + m_cosmo.f0 * B * cos_mx;
        const amrex::Real psi_reg2 = a_psi_reg * a_psi_reg;
        const amrex::Real psi_reg6 = psi_reg2 * psi_reg2 * psi_reg2;
        out.Pi = B * Q_x / psi_reg6 + (1.0 - B) * m_derived.Pi_BH;
        out.V  = m_potential.V_of_phi(out.phi);
        return out;
    }

    //! Free traceless part B Ahat^Xue (diagonal) and its flat divergence
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE void
    Ahat_free(int ix, const Tensor::Rank1 &a_displacement,
              Tensor::Rank1 &a_diagonal, Tensor::Rank1 &a_divergence) const
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        amrex::Real B{};
        Tensor::Rank1 d1_B{};
        homogenisation(a_displacement, B, d1_B);

        const amrex::Real m     = m_derived.wavenumber;
        const amrex::Real phase = m * x_coord(ix);
        const amrex::Real k2f1  = m_derived.kappa2 * m_cosmo.f1;
        const amrex::Real A11 =
            k2f1 * (-m_derived.Pi0 * std::cos(phase) -
                    0.25 * m_cosmo.f0 * std::cos(2.0 * phase));
        const amrex::Real dA11_dx = k2f1 * m *
                                    (m_derived.Pi0 * std::sin(phase) +
                                     0.5 * m_cosmo.f0 * std::sin(2.0 * phase));

        const std::array<amrex::Real, 3> factor{1.0, m_cosmo.lambda,
                                                -(1.0 + m_cosmo.lambda)};
        FOR (idir)
        {
            a_diagonal(idir) = B * factor[idir] * A11;
            // d_j (B A_ij) = B d_j A_ij + A_ij d_j B, A diagonal in x only
            a_divergence(idir) = factor[idir] * A11 * d1_B(idir);
        }
        a_divergence(0) += B * dA11_dx;
    }

    [[nodiscard]] AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
    x_coord(int ix) const
    {
        amrex::Real x{};
        Coordinates::compute_coord(x, ix, m_dx);
        return x;
    }

    [[nodiscard]] const derived_t &get_derived() const { return m_derived; }
    [[nodiscard]] const bh_params_t &get_bh_params() const { return m_bh; }
    [[nodiscard]] const Potential &get_potential() const { return m_potential; }

  protected:
    //! C^4 step S(s) = s^5 (126 - 420 s + 540 s^2 - 315 s^3 + 70 s^4),
    //! S'(s) = 630 s^4 (1 - s)^4, of r between a_inner and a_outer, and its
    //! first two derivatives with respect to r. (A C^2 step leaves a jump in
    //! the third derivative of Laplacian(f), which shows up as spikes in the
    //! finite-difference BSSN constraints at the window edges.)
    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE static void
    smooth_step(amrex::Real a_r, amrex::Real a_inner, amrex::Real a_outer,
                amrex::Real &a_step, amrex::Real &a_d_step,
                amrex::Real &a_d2_step)
    // NOLINTEND(bugprone-easily-swappable-parameters)
    {
        a_d_step  = 0.0;
        a_d2_step = 0.0;
        if (a_r <= a_inner)
        {
            a_step = 0.0;
            return;
        }
        if (a_r >= a_outer)
        {
            a_step = 1.0;
            return;
        }
        const amrex::Real width = a_outer - a_inner;
        const amrex::Real s     = (a_r - a_inner) / width;
        const amrex::Real s2    = s * s;
        const amrex::Real t     = 1.0 - s;
        const amrex::Real t2    = t * t;
        a_step =
            s2 * s2 * s *
            (126.0 - 420.0 * s + 540.0 * s2 - 315.0 * s2 * s + 70.0 * s2 * s2);
        a_d_step = 630.0 * s2 * s2 * t2 * t2 / width;
        a_d2_step =
            2520.0 * s2 * s * t2 * t * (1.0 - 2.0 * s) / (width * width);
    }

    amrex::Real m_dx;
    bh_params_t m_bh{};
    cosmo_params_t m_cosmo{};
    Potential m_potential;
    derived_t m_derived{};
};

inline void EkpyroticBHData::bh_params_t::check_params()
{
    GRParmParse pp("bh");

    amrex::Real mass{0.0};
    pp.queryAdd("mass", mass);
    if (mass < 0.0)
    {
        pp.error("mass", "must be >= 0.0");
    }

    GRParmParse geometry_pp("geometry");
    std::array<amrex::Real, AMREX_SPACEDIM> domain_lengths{};
    geometry_pp.get("prob_extent", domain_lengths);
    std::array<int, AMREX_SPACEDIM> is_periodic{};
    geometry_pp.get("is_periodic", is_periodic);
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
    {
        if (is_periodic[idir] != 1)
        {
            geometry_pp.error("is_periodic", "the ekpyrotic black hole "
                                             "initial data require periodic "
                                             "boundary conditions");
        }
    }

    std::array<amrex::Real, AMREX_SPACEDIM> center{};
    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
    {
        center[idir] = 0.5 * domain_lengths[idir];
    }
    pp.queryAdd("center", center);

    const amrex::Real half_min_length =
        0.5 * *std::min_element(domain_lengths.begin(), domain_lengths.end());
    amrex::Real window_inner{0.25 * half_min_length};
    amrex::Real window_outer{0.9 * half_min_length};
    pp.queryAdd("window_inner", window_inner);
    pp.queryAdd("window_outer", window_outer);
    amrex::Real homogeneous_inner{0.25 * half_min_length};
    amrex::Real homogeneous_outer{0.5 * half_min_length};
    pp.queryAdd("homogeneous_inner", homogeneous_inner);
    pp.queryAdd("homogeneous_outer", homogeneous_outer);

    if (mass > 0.0)
    {
        if (!(0.0 < window_inner && window_inner < window_outer &&
              window_outer < half_min_length))
        {
            pp.error("window_outer", "need 0 < window_inner < window_outer < "
                                     "half the shortest side of the box");
        }
        if (!(0.0 < homogeneous_inner &&
              homogeneous_inner < homogeneous_outer &&
              homogeneous_outer < half_min_length))
        {
            pp.error("homogeneous_outer",
                     "need 0 < homogeneous_inner < homogeneous_outer < half "
                     "the shortest side of the box");
        }

        // The puncture must not sit on a cell centre (r = 0)
        GRParmParse amr_pp("amr");
        std::array<int, AMREX_SPACEDIM> n_cell{};
        amr_pp.get("n_cell", n_cell);
        for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
        {
            const amrex::Real dx       = domain_lengths[idir] / n_cell[idir];
            const amrex::Real in_cells = center[idir] / dx;
            if (std::abs(in_cells - std::round(in_cells)) > 1.0e-8)
            {
                pp.warning("center", "is not on a vertex of the coarsest "
                                     "grid; the puncture may lie close to a "
                                     "cell centre");
            }
        }
    }
}

inline void EkpyroticBHData::cosmo_params_t::check_params()
{
    GRParmParse pp("initial_data");

    amrex::Real phi0{0.0};
    amrex::Real Pi0{0.0};
    bool use_scaling_Pi0{true};
    amrex::Real f0_value{0.0};
    amrex::Real f1_value{0.0};
    amrex::Real lambda{0.0};
    int mode_number{1};

    pp.queryAdd("phi0", phi0);
    pp.queryAdd("Pi0", Pi0);
    pp.queryAdd("use_scaling_Pi0", use_scaling_Pi0);
    pp.queryAdd("f0", f0_value);
    pp.queryAdd("f1", f1_value);
    pp.queryAdd("lambda", lambda);
    pp.queryAdd("mode_number", mode_number);

    if (mode_number < 1)
    {
        pp.error("mode_number", "must be >= 1");
    }

    bh_params_t bh;
    bh.fill_params();
    cosmo_params_t cosmo;
    cosmo.fill_params();
    const Potential potential;
    const derived_t derived = compute_derived(bh, cosmo, potential);

    if (use_scaling_Pi0 &&
        scaling_Pi0_squared(derived.kappa2, potential, phi0) < 0.0)
    {
        pp.error("use_scaling_Pi0", "the ekpyrotic scaling solution requires "
                                    "2 c^2 V(phi0) / (6 kappa^2 - c^2) >= 0");
    }

    if (derived.mean_rho <= 0.0)
    {
        amrex::Abort("EkpyroticBHData: the average energy density <rho> = " +
                     std::to_string(derived.mean_rho) +
                     " is not positive, so K0 is not defined");
    }

    amrex::Print() << "EkpyroticBHData: kappa^2 = " << derived.kappa2
                   << ", Pi0 = " << derived.Pi0
                   << ", m = " << derived.wavenumber
                   << ", <rho> = " << derived.mean_rho << ", M = " << bh.mass
                   << ", V_box = " << derived.box_volume
                   << ", K0 = " << derived.K0 << ", Pi_BH = " << derived.Pi_BH
                   << ", dtK = " << derived.dtK << "\n";
}

#endif /* EKPYROTICBHDATA_HPP_ */
