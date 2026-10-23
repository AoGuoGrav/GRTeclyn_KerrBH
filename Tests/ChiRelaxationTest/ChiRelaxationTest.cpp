/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */
// Doctest header
#include "doctest.h"

// Test header
#include "ChiRelaxationTest.hpp"

// Common test headers
#include "InitialData.hpp"
#include "doctestCLIArgs.hpp"

// GRTeclyn headers
#include "ChiHyperbolicRelaxation.hpp"
#include "ChiRelaxation.hpp"
#include "Constraints.hpp"
#include "DefaultPotential.hpp"
#include "EMTensor.hpp"
#include "ScalarField.hpp"

// AMReX headers
#include "AMReX.H"
#include "AMReX_MultiFab.H"

// System headers
#include <cmath>

void run_chi_relaxation_test()
{
    int amrex_argc    = doctest::cli_args.argc();
    char **amrex_argv = doctest::cli_args.argv();

    // NOLINTNEXTLINE(bugprone-casting-through-void) // Open MPI triggers this
    amrex::Initialize(amrex_argc, amrex_argv);
    {
        constexpr int num_cells  = 32;
        constexpr int num_ghosts = 3;
        constexpr double dx      = 0.25 / (num_cells - 1);

        amrex::Box box(
            amrex::IntVect(0, 0, 0),
            amrex::IntVect(num_cells - 1, num_cells - 1, num_cells - 1));

        amrex::BoxArray box_array{box};
        amrex::RealVect dx_Vect{dx};
        amrex::RealBox real_box{box, dx_Vect.dataPtr(),
                                amrex::RealVect::Zero.dataPtr()};
        int coord_sys = 0; // Cartesian
        amrex::Geometry geom{box, &real_box, coord_sys};
        amrex::DistributionMapping distribution_mapping{box_array};
        amrex::MFInfo mf_info;
        mf_info.SetArena(amrex::The_Managed_Arena());

        amrex::MultiFab in_mf{box_array, distribution_mapping, NUM_VARS,
                              num_ghosts, mf_info};

        const auto &in_array = in_mf.arrays();

        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        amrex::ParallelFor(
            in_mf, in_mf.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int ibox, int i, int j, int k)
            // NOLINTEND(bugprone-easily-swappable-parameters)
            {
                const amrex::IntVect iv{i, j, k};
                const amrex::RealVect coords = amrex::RealVect{iv} * dx;

                random_ccz4_initial_data(iv, in_array[ibox], coords);
                random_matter_bssn_initial_data(iv, in_array[ibox], coords);
            });

        using DefaultScalarField = ScalarField<DefaultPotential>;

        amrex::Real G_Newton    = 1.0;
        amrex::Real relax_speed = 0.1;

        ChiRelaxation<DefaultScalarField> chi_relaxation{dx, relax_speed,
                                                         G_Newton};

        amrex::MultiFab out_mf{box_array, distribution_mapping, NUM_VARS, 0,
                               mf_info};

        const auto &in_c_array   = in_mf.const_arrays();
        const auto &out_mf_array = out_mf.arrays();

        amrex::ParallelFor(out_mf,
                           [=] AMREX_GPU_DEVICE(int ibox, int i, int j, int k)
                           {
                               chi_relaxation.compute(i, j, k,
                                                      out_mf_array[ibox],
                                                      in_c_array[ibox]);
                           });

        // GPU barrier
        amrex::Gpu::streamSynchronize();

        CHECK(!out_mf.contains_nan());

        // The GRChombo comparison data (GRChomboChiRelaxationTest.h5) from the
        // enhancement/chi_relaxation branch do not correspond to the initial
        // data above, so instead check that
        //     rhs_chi = speed * (Ham_vacuum - 16 pi G rho) / chi,
        // where the vacuum constraints and rho are checked against GRChombo in
        // the Constraints and EMTensor tests
        amrex::Real time = 0.0;
        int *bcrec       = nullptr;
        int level        = 0;

        constexpr int num_constraints = 4;
        amrex::MultiFab constraints_mf{box_array, distribution_mapping,
                                       num_constraints, 0, mf_info};
        Constraints::compute_mf(constraints_mf, 0, num_constraints, in_mf, geom,
                                time, bcrec, level);

        using EnergyDensity =
            EMTensor<DefaultScalarField, EMTensorOptions::justEnergyDensity>;
        amrex::MultiFab rho_mf{box_array, distribution_mapping, 1, 0, mf_info};
        EnergyDensity::compute_mf(rho_mf, 0, 1, in_mf, geom, time, bcrec,
                                  level);

        amrex::MultiFab diff_mf{box_array, distribution_mapping, 1, 0, mf_info};
        const auto &diff_array        = diff_mf.arrays();
        const auto &constraints_array = constraints_mf.const_arrays();
        const auto &rho_array         = rho_mf.const_arrays();

        amrex::ParallelFor(
            diff_mf,
            [=] AMREX_GPU_DEVICE(int ibox, int i, int j, int k)
            {
                const amrex::Real Ham =
                    constraints_array[ibox](i, j, k, 0) -
                    16.0 * M_PI * G_Newton * rho_array[ibox](i, j, k);
                diff_array[ibox](i, j, k) =
                    out_mf_array[ibox](i, j, k, c_chi) -
                    relax_speed * Ham / in_c_array[ibox](i, j, k, c_chi);
            });

        // GPU barrier
        amrex::Gpu::streamSynchronize();

        // All the other RHS are zero
        for (int icomp = 0; icomp < NUM_VARS; ++icomp)
        {
            if (icomp != c_chi)
            {
                CHECK(out_mf.norm0(icomp) == 0.0);
            }
        }

        INFO("max |rhs_chi| = " << out_mf.norm0(c_chi));
        CHECK(out_mf.norm0(c_chi) > 0.0);
        CHECK(diff_mf.norm0(0) < 1e-10 * out_mf.norm0(c_chi));
    }
    amrex::Finalize();
}

void run_chi_hyperbolic_relaxation_test()
{
    int amrex_argc    = doctest::cli_args.argc();
    char **amrex_argv = doctest::cli_args.argv();

    // NOLINTNEXTLINE(bugprone-casting-through-void) // Open MPI triggers this
    amrex::Initialize(amrex_argc, amrex_argv);
    {
        constexpr int num_cells  = 32;
        constexpr int num_ghosts = 3;
        constexpr double dx      = 0.25 / (num_cells - 1);

        amrex::Box box(
            amrex::IntVect(0, 0, 0),
            amrex::IntVect(num_cells - 1, num_cells - 1, num_cells - 1));

        amrex::BoxArray box_array{box};
        amrex::DistributionMapping distribution_mapping{box_array};
        amrex::MFInfo mf_info;
        mf_info.SetArena(amrex::The_Managed_Arena());

        // The test StateVariables have no theta, so use an unused matter
        // component for it
        constexpr int c_test_theta = c_phi_Re;

        amrex::MultiFab in_mf{box_array, distribution_mapping, NUM_VARS,
                              num_ghosts, mf_info};
        const auto &in_array = in_mf.arrays();

        // NOLINTBEGIN(bugprone-easily-swappable-parameters)
        amrex::ParallelFor(
            in_mf, in_mf.nGrowVect(),
            [=] AMREX_GPU_DEVICE(int ibox, int i, int j, int k)
            // NOLINTEND(bugprone-easily-swappable-parameters)
            {
                const amrex::IntVect iv{i, j, k};
                const amrex::RealVect coords = amrex::RealVect{iv} * dx;

                random_ccz4_initial_data(iv, in_array[ibox], coords);
                random_matter_bssn_initial_data(iv, in_array[ibox], coords);
                in_array[ibox](i, j, k, c_test_theta) =
                    0.3 * std::sin(2.0 * coords[0]) +
                    0.1 * std::cos(3.0 * coords[1] - coords[2]);
            });

        using DefaultScalarField = ScalarField<DefaultPotential>;

        const amrex::Real G_Newton      = 1.0;
        const amrex::Real coefficient_A = 7.0;
        const amrex::Real eta_damping   = 1.5;

        // Reference: the parabolic relaxation with unit speed gives Ham / chi
        const ChiRelaxation<DefaultScalarField> parabolic{dx, 1.0, G_Newton};
        const ChiHyperbolicRelaxation<DefaultScalarField> hyperbolic{
            dx, coefficient_A, c_test_theta, eta_damping, G_Newton};

        amrex::MultiFab ref_mf{box_array, distribution_mapping, NUM_VARS, 0,
                               mf_info};
        amrex::MultiFab out_mf{box_array, distribution_mapping, NUM_VARS, 0,
                               mf_info};
        amrex::MultiFab diff_mf{box_array, distribution_mapping, 2, 0, mf_info};

        const auto &in_c_array = in_mf.const_arrays();
        const auto &ref_array  = ref_mf.arrays();
        const auto &out_array  = out_mf.arrays();
        const auto &diff_array = diff_mf.arrays();

        amrex::ParallelFor(
            out_mf,
            [=] AMREX_GPU_DEVICE(int ibox, int i, int j, int k)
            {
                parabolic.compute(i, j, k, ref_array[ibox], in_c_array[ibox]);
                hyperbolic.compute(i, j, k, out_array[ibox], in_c_array[ibox]);

                const amrex::Real ham_over_chi =
                    ref_array[ibox](i, j, k, c_chi);
                const amrex::Real theta =
                    in_c_array[ibox](i, j, k, c_test_theta);
                diff_array[ibox](i, j, k, 0) =
                    out_array[ibox](i, j, k, c_chi) - theta;
                diff_array[ibox](i, j, k, 1) =
                    out_array[ibox](i, j, k, c_test_theta) -
                    (coefficient_A * ham_over_chi - eta_damping * theta);
            });

        // GPU barrier
        amrex::Gpu::streamSynchronize();

        CHECK(!out_mf.contains_nan());

        // All the other RHS are zero
        for (int icomp = 0; icomp < NUM_VARS; ++icomp)
        {
            if (icomp != c_chi && icomp != c_test_theta)
            {
                CHECK(out_mf.norm0(icomp) == 0.0);
            }
        }

        INFO("max |rhs_chi| = " << out_mf.norm0(c_chi) << ", max |rhs_theta| = "
                                << out_mf.norm0(c_test_theta));
        CHECK(out_mf.norm0(c_chi) > 0.0);
        CHECK(out_mf.norm0(c_test_theta) > 0.0);
        // rhs_chi is a copy of theta, so it must agree exactly
        CHECK(diff_mf.norm0(0) == 0.0);
        CHECK(diff_mf.norm0(1) < 1e-12 * out_mf.norm0(c_test_theta));
    }
    amrex::Finalize();
}
