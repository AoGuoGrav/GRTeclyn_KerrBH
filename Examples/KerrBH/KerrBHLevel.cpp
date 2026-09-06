/* GRTeclyn
 *
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#include "KerrBHLevel.hpp"

#include "AlgebraicConstraintsEnforcer.hpp"
#include "CCZ4RHS.hpp"
#include "ChiTagger.hpp"
#include "Constraints.hpp"
#include "ExtractionTagger.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GammaCalculator.hpp"
#include "GRParmParse.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
#include "KerrBHInitialData.hpp"
#include "PositiveChiAndLapse.hpp"
#include "PunctureTagger.hpp"
#include "PunctureTracker.hpp"
#include "SixthOrderDerivatives.hpp"
#include "Weyl4.hpp"
#include "WeylExtraction.hpp"

#include <array>
#include <cstddef>
#include <type_traits>

BHAmr<KerrBHLevel::num_punctures> *KerrBHLevel::get_bh_amr_ptr()
{
    return dynamic_cast<BHAmr<num_punctures> *>(get_gr_amr_ptr());
}

PunctureTracker<KerrBHLevel::num_punctures> &
KerrBHLevel::get_puncture_tracker()
{
    return get_bh_amr_ptr()->get_puncture_tracker();
}

void KerrBHLevel::variableSetUp()
{
    BL_PROFILE("KerrBHLevel::variableSetUp()");

    // Set up the state variables.
    state_variable_set_up();

    Constraints::set_up(state_index);

    Weyl4::set_up(state_index);
}

// Things to do during the advance step after RK4 steps.
void KerrBHLevel::specific_advance()
{
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays = state_new.arrays();

    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    PositiveChiAndLapse positive_chi_lapse;

    // Enforce det(h)=1, the trace-free A_ij condition, and positive chi
    // and lapse.
    amrex::ParallelFor(
        state_new,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            algebraic_constraints_enforcer(ix, iy, iz,
                                           state_arrays[box_no]);
            positive_chi_lapse(ix, iy, iz, state_arrays[box_no]);
        });
}

void KerrBHLevel::initData()
{
    BL_PROFILE("KerrBHLevel::initData");

    if (get_gr_amr_ptr()->Verbose() > 0)
    {
        amrex::Print() << "KerrBHLevel::initData " << Level() << "\n";
    }

    // Construct the analytic semi-isotropic Kerr initial data.
    const amrex::Real dx = Geom().CellSize(0);

    const KerrBHInitialData kerr_initial_data(dx);

    static_assert(std::is_trivially_copyable_v<KerrBHInitialData>,
                  "KerrBHInitialData must be device copyable");

    // Zero every state component first. KerrBHInitialData then overwrites
    // chi, h_ij, K, A_ij, lapse, and shift with analytic Kerr values.
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays = state_new.arrays();

    amrex::ParallelFor(
        state_new, state_new.nGrowVect(),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            amrex::CellData<amrex::Real> cell =
                state_arrays[box_no].cellData(ix, iy, iz);

            for (int n = 0; n < cell.nComp(); ++n)
            {
                cell[n] = 0.0;
            }

            kerr_initial_data(ix, iy, iz, state_arrays[box_no]);
        });

    // Compute Gamma^i using the same finite-difference order as the RHS.
    // Then initialise B^i = Gamma^i for the integrated Gamma-driver gauge.
    if (m_evolution_spatial_derivative_order == 4)
    {
        const GammaCalculator<FourthOrderDerivatives> gamma_calculator(dx);

        const IntegratedMovingPunctureGauge<FourthOrderDerivatives>
            integrated_moving_puncture_gauge(dx);

        amrex::ParallelFor(
            state_new,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                gamma_calculator(ix, iy, iz, state_arrays[box_no]);
            });

        amrex::ParallelFor(
            state_new,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                integrated_moving_puncture_gauge.set_initial_B_to_Gamma(
                    ix, iy, iz, state_arrays[box_no]);
            });
    }
    else if (m_evolution_spatial_derivative_order == 6)
    {
        const GammaCalculator<SixthOrderDerivatives> gamma_calculator(dx);

        const IntegratedMovingPunctureGauge<SixthOrderDerivatives>
            integrated_moving_puncture_gauge(dx);

        amrex::ParallelFor(
            state_new,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                gamma_calculator(ix, iy, iz, state_arrays[box_no]);
            });

        amrex::ParallelFor(
            state_new,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                integrated_moving_puncture_gauge.set_initial_B_to_Gamma(
                    ix, iy, iz, state_arrays[box_no]);
            });
    }

    amrex::Gpu::streamSynchronize();

    // A single Kerr BH has one puncture at its configured centre.
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        KerrBHInitialData::params_t kerr_params;
        kerr_params.fill_params();

        get_puncture_tracker().set_puncture_coords(
            {kerr_params.center[0], kerr_params.center[1],
             kerr_params.center[2]});

        // We cannot call start_from_initial_punctures() here because the
        // complete AMR grid is not available yet.
    }
}

// Calculate RHS during RK4 substeps.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void KerrBHLevel::specific_eval_rhs(amrex::MultiFab &a_soln,
                                    amrex::MultiFab &a_rhs,
                                    const amrex::Real /*a_time*/)
{
    BL_PROFILE("KerrBHLevel::specific_eval_rhs()");

    const auto &soln_arrays = a_soln.arrays();
    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays = a_rhs.arrays();
    const auto soln_ghosts = a_soln.nGrowVect();

    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    PositiveChiAndLapse positive_chi_lapse;

    // Enforce positive chi and lapse, det(h)=1, and trace-free A_ij.
    amrex::ParallelFor(
        a_soln, soln_ghosts,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            algebraic_constraints_enforcer(ix, iy, iz,
                                           soln_arrays[box_no]);
            positive_chi_lapse(ix, iy, iz, soln_arrays[box_no]);
        });

    // Calculate CCZ4 right hand side using the selected derivative order.
    if (m_evolution_spatial_derivative_order == 4)
    {
        CCZ4RHS<FourthOrderDerivatives> ccz4rhs(Geom().CellSize(0));

        IntegratedMovingPunctureGauge<FourthOrderDerivatives>
            moving_puncture_gauge(Geom().CellSize(0));

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_chi_and_h_ij(
                    ix, iy, iz, rhs_arrays[box_no],
                    const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_A_ij_and_Theta_and_Gamma(
                    ix, iy, iz, rhs_arrays[box_no],
                    const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                moving_puncture_gauge.calculate_rhs(
                    ix, iy, iz, rhs_arrays[box_no],
                    const_soln_arrays[box_no]);

                ccz4rhs.apply_dissipation(
                    ix, iy, iz, rhs_arrays[box_no],
                    const_soln_arrays[box_no]);
            });
    }
    else if (m_evolution_spatial_derivative_order == 6)
    {
        CCZ4RHS<SixthOrderDerivatives> ccz4rhs(Geom().CellSize(0));

        IntegratedMovingPunctureGauge<SixthOrderDerivatives>
            moving_puncture_gauge(Geom().CellSize(0));

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_chi_and_h_ij(
                    ix, iy, iz, rhs_arrays[box_no],
                    const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                ccz4rhs.compute_A_ij_and_Theta_and_Gamma(
                    ix, iy, iz, rhs_arrays[box_no],
                    const_soln_arrays[box_no]);
            });

        amrex::ParallelFor(
            a_rhs,
            [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            {
                moving_puncture_gauge.calculate_rhs(
                    ix, iy, iz, rhs_arrays[box_no],
                    const_soln_arrays[box_no]);

                ccz4rhs.apply_dissipation(
                    ix, iy, iz, rhs_arrays[box_no],
                    const_soln_arrays[box_no]);
            });
    }

    amrex::Gpu::streamSynchronize();
}

// Enforce algebraic constraints during RK4 substeps.
void KerrBHLevel::specific_update_ode(amrex::MultiFab &a_soln)
{
    AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const auto soln_ghosts = amrex::IntVect(0);

    // Enforce det(h)=1 and the trace-free A_ij condition.
    const auto &soln_arrays = a_soln.arrays();

    amrex::ParallelFor(
        a_soln, soln_ghosts,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            algebraic_constraints_enforcer(ix, iy, iz,
                                           soln_arrays[box_no]);
        });

    amrex::Gpu::streamSynchronize();
}

void KerrBHLevel::pre_tag_cells()
{
    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto current_time = get_state_data(state_index).curTime();

    // Fill ghosts for chi to calculate second derivatives.
    // A fourth-order second derivative requires two ghost cells.
    const int num_ghosts = 2;
    const int num_comps = 1;

    FillPatch(*this, state_new, num_ghosts, current_time, state_index,
              c_chi, num_comps);
}

void KerrBHLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array,
                            amrex::Real a_regrid_threshold)
{
    BL_PROFILE("KerrBHLevel::tag_cells()");

    amrex::MultiFab &state_new = get_new_data(state_index);

    const auto &tag_arrays = a_tag_box_array.arrays();
    const auto &state_const_arrays = state_new.const_arrays();

    ChiTagger chi_tagger(Geom().CellSize(0), a_regrid_threshold);

    spherical_extraction_params_t extraction_params("weyl_extraction");
    extraction_params.fill_params();

    ExtractionTagger extraction_tagger(Geom().CellSize(0), Level(),
                                       extraction_params);

    constexpr auto num_puncture_coords =
        static_cast<std::size_t>(AMREX_SPACEDIM * num_punctures);

    std::array<amrex::Real, num_puncture_coords> puncture_coords{};

    const bool puncture_tracking_enabled =
        get_bh_amr_ptr()->puncture_tracking_enabled();

    if (puncture_tracking_enabled)
    {
        puncture_coords = get_puncture_tracker().get_puncture_coords();
    }

    KerrBHInitialData::params_t kerr_params;
    kerr_params.fill_params();

    const std::array<amrex::Real, num_punctures> puncture_masses{
        kerr_params.mass};

    PunctureTagger<num_punctures> puncture_tagger(
        Geom().CellSize(0), Level(), get_gr_amr_ptr()->maxLevel(),
        puncture_coords, puncture_masses);

    amrex::ParallelFor(
        state_new, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            chi_tagger(ix, iy, iz, tag_arrays[box_no],
                       state_const_arrays[box_no]);

            extraction_tagger(ix, iy, iz, tag_arrays[box_no]);

            if (puncture_tracking_enabled)
            {
                puncture_tagger(ix, iy, iz, tag_arrays[box_no]);
            }
        });

    amrex::Gpu::streamSynchronize();
}

void KerrBHLevel::specific_post_init()
{
    BL_PROFILE("KerrBHLevel::specific_post_init()");

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().start_from_initial_punctures();
    }
}

void KerrBHLevel::specific_post_restart()
{
    BL_PROFILE("KerrBHLevel::specific_post_restart()");

    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        std::string restart_checkpoint{};

        GRParmParse pp("amr");
        pp.get("restart", restart_checkpoint);

        get_puncture_tracker().restart(restart_checkpoint);
    }
}

void KerrBHLevel::specific_post_plotfile(const std::string &a_dir,
                                         std::ostream & /*a_os*/)
{
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().write_plotfile(a_dir);
    }
}

void KerrBHLevel::specific_post_checkpoint(
    const std::string &a_chk_dir, std::ostream & /*a_os*/)
{
    if (get_bh_amr_ptr()->puncture_tracking_enabled() && Level() == 0)
    {
        get_puncture_tracker().checkpoint(a_chk_dir);
    }
}

void KerrBHLevel::specific_post_timestep()
{
    BL_PROFILE("KerrBHLevel::specific_post_timestep");

    if (get_bh_amr_ptr()->puncture_tracking_enabled())
    {
        GRParmParse puncture_tracking_pp("puncture_tracking");

        int puncture_tracking_level{};
        puncture_tracking_pp.get("level", puncture_tracking_level);

        int puncture_tracking_writeout_level{};
        puncture_tracking_pp.get("writeout_level",
                                 puncture_tracking_writeout_level);

        // Do puncture tracking on the requested level.
        if (Level() == puncture_tracking_level)
        {
            BL_PROFILE("PunctureTracking");

            // Only write punctures at requested timestep multiples.
            const bool write_punctures =
                at_level_timestep_multiple(puncture_tracking_writeout_level);

            const amrex::Real current_time =
                get_state_data(state_index).curTime();

            const amrex::Real dt =
                get_gr_amr_ptr()->dtLevel(Level());

            get_puncture_tracker().track(current_time, dt,
                                         write_punctures);
        }
    }

    spherical_extraction_params_t extraction_params("weyl_extraction");
    extraction_params.fill_params();

    if (extraction_params.enabled)
    {
        const int min_level =
            extraction_params.min_extraction_level();

        const bool calculate_weyl =
            at_level_timestep_multiple(min_level);

        if (calculate_weyl && Level() == min_level)
        {
            const amrex::Real m_time =
                get_state_data(state_index).curTime();

            const amrex::Real m_dt =
                get_gr_amr_ptr()->dtLevel(Level());

            const amrex::Real restart_time =
                get_gr_amr_ptr()->get_restart_time();

            const bool first_step = (m_time <= m_dt);

            WeylExtraction my_extraction(
                extraction_params, m_dt, m_time, first_step, restart_time);

            my_extraction.execute_query(
                &get_bh_amr_ptr()->m_weyl_interpolator);
        }
    }
}