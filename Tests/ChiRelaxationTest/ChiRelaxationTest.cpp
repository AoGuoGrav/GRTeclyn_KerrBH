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
#include "ChiRelaxation.hpp"
#include "Constraints.hpp"
#include "DefaultPotential.hpp"
#include "EMTensor.hpp"
#include "ScalarField.hpp"

// AMReX headers
#include "AMReX.H"
#include "AMReX_FArrayBox.H"
#include "AMReX_MultiFab.H"

#ifdef AMREX_USE_HDF5
#include <AMReX_PlotFileUtilHDF5.H>
#endif

// System headers
#include <array>
#include <cstdlib>
#include <iostream>
#include <string>

void run_chi_relaxation_test()
{
    int amrex_argc    = doctest::cli_args.argc();
    char **amrex_argv = doctest::cli_args.argv();

    // NOLINTNEXTLINE(bugprone-casting-through-void)
    amrex::Initialize(amrex_argc, amrex_argv);
    {

        constexpr int num_cells  = 32;
        constexpr int num_ghosts = 3;
        constexpr double dx      = 0.25 / (num_cells - 1);

        amrex::Box box(
            amrex::IntVect(0, 0, 0),
            amrex::IntVect(num_cells - 1, num_cells - 1, num_cells - 1));

        amrex::Box ghosted_box = box;
        ghosted_box.grow(num_ghosts);

        amrex::BoxArray box_array{box};
        amrex::RealVect dx_Vect{dx};
        amrex::RealBox real_box{box, dx_Vect.dataPtr(),
                                amrex::RealVect::Zero.dataPtr()};
        int coord_sys = 0;
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
                amrex::Real x                = coords[0];
                amrex::Real y                = coords[1];
                amrex::Real z                = coords[2];

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
        // data above, so instead check that rhs_chi = speed * (Ham_vacuum -
        // 16 pi G rho) / chi, where the vacuum constraints and rho are checked
        // against GRChombo in the Constraints and EMTensor tests
        amrex::MultiFab constraints_mf{box_array, distribution_mapping, 4, 0,
                                       mf_info};
        Constraints::compute_mf(constraints_mf, 0, 4, in_mf, geom, 0.0,
                                nullptr, 0);

        amrex::MultiFab rho_mf{box_array, distribution_mapping, 1, 0, mf_info};
        EMTensor<DefaultScalarField,
                 EMTensorOptions::justEnergyDensity>::compute_mf(rho_mf, 0, 1,
                                                                 in_mf, geom,
                                                                 0.0, nullptr,
                                                                 0);

        amrex::MultiFab diff_mf{box_array, distribution_mapping, 1, 0,
                                mf_info};
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
