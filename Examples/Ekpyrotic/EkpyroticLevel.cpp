/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#include "EkpyroticLevel.hpp"

#include "AlgebraicConstraintsEnforcer.hpp"
#include "AnisotropyDiagnostic.hpp"
#include "CCZ4RHSWithMatter.hpp"
#include "ChiRelaxationParameters.hpp"
#include "ConstraintsWithMatter.hpp"
#include "DiagnosticsParameters.hpp"
#include "EkpyroticInitialData.hpp"
#include "FixedGridsTagger.hpp"
#include "FourthOrderDerivatives.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
#include "Interval.hpp"
#include "LineExtraction.hpp"
#include "PositiveChiAndLapse.hpp"
#include "SixthOrderDerivatives.hpp"
#include "StateTypes.hpp"

#include <cmath>
#include <iomanip>
#include <type_traits>

using EkpyroticConstraints =
    ConstraintsWithMatter<EkpyroticLevel::ScalarFieldWithPotential<>>;

EkpyroticAmr *EkpyroticLevel::get_ekpyrotic_amr_ptr()
{
    return dynamic_cast<EkpyroticAmr *>(get_gr_amr_ptr());
}

void EkpyroticLevel::variableSetUp()
{
    BL_PROFILE("EkpyroticLevel::variableSetUp()");
    state_variable_set_up();
    EkpyroticConstraints::set_up(state_index);
    AnisotropyDiagnostic::set_up(state_index);
}

void EkpyroticLevel::specific_advance()
{
    BL_PROFILE("EkpyroticLevel::specific_advance()");

    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const PositiveChiAndLapse positive_chi_and_lapse;

    amrex::ParallelFor(
        state_new,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            algebraic_constraints_enforcer(ix, iy, iz, state_arrays[box_no]);
            positive_chi_and_lapse(ix, iy, iz, state_arrays[box_no]);
        });

    // If this step was a relaxation step, make Pi and A_ij consistent with
    // the new chi so that the momentum constraint holds exactly
    chi_relaxation_params_t relaxation;
    relaxation.fill_params();
    const amrex::Real step_start_time = get_state_data(state_index).prevTime();
    if (step_start_time < relaxation.time)
    {
        const EkpyroticInitialData initial_data(Geom().CellSize(0));
        amrex::ParallelFor(
            state_new, [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
            { initial_data.set_momenta(ix, iy, iz, state_arrays[box_no]); });
    }
    amrex::Gpu::streamSynchronize();
}

void EkpyroticLevel::specific_post_init()
{
    BL_PROFILE("EkpyroticLevel::specific_post_init()");
    if (Level() == 0)
    {
        print_diagnostics();
    }
}

void EkpyroticLevel::specific_post_timestep()
{
    BL_PROFILE("EkpyroticLevel::specific_post_timestep()");

    if (Level() != 0)
    {
        return;
    }

    diagnostics_params_t diagnostics;
    diagnostics.fill_params();
    const int step = get_gr_amr_ptr()->levelSteps(0);

    if (step % diagnostics.print_interval == 0)
    {
        print_diagnostics();
    }

    if (step % diagnostics.extraction_interval != 0)
    {
        return;
    }

    const amrex::Real time         = get_state_data(state_index).curTime();
    const amrex::Real dt           = get_gr_amr_ptr()->dtLevel(0);
    const amrex::Real restart_time = get_gr_amr_ptr()->get_restart_time();
    // The first extraction after (re)start creates the output directory
    const bool first_step =
        (time < restart_time + (diagnostics.extraction_interval + 0.5) * dt);
    EkpyroticAmr *amr_ptr = get_ekpyrotic_amr_ptr();

    const LineExtraction<1> chi_extraction("chi_line_extraction", c_chi, dt,
                                           time, restart_time, first_step);
    chi_extraction.execute_query(&amr_ptr->chi_interpolator);

    const LineExtraction<1> K_extraction("K_line_extraction", c_K, dt, time,
                                         restart_time, first_step);
    K_extraction.execute_query(&amr_ptr->K_interpolator);

    const LineExtraction<1> lapse_extraction(
        "lapse_line_extraction", c_lapse, dt, time, restart_time, first_step);
    lapse_extraction.execute_query(&amr_ptr->lapse_interpolator);

    const LineExtraction<2> scalar_field_extraction(
        "scalar_field_line_extraction", c_phi, dt, time, restart_time,
        first_step);
    scalar_field_extraction.execute_query(&amr_ptr->scalar_field_interpolator);

    const LineExtraction<4> constraints_extraction(
        "constraints_line_extraction", 0, dt, time, restart_time, first_step);
    const LineExtraction<4>::derived_vars_t constraints_vars{
        Constraints::name,
        {"Ham",          "Mom1",          "Mom2",          "Mom3"         },
        {BCParity::even, BCParity::odd_x, BCParity::odd_y, BCParity::odd_z}
    };
    constraints_extraction.execute_query(&amr_ptr->constraints_interpolator,
                                         constraints_vars);

    const LineExtraction<3> anisotropy_extraction(
        "anisotropy_line_extraction", 0, dt, time, restart_time, first_step);
    const LineExtraction<3>::derived_vars_t anisotropy_vars{
        AnisotropyDiagnostic::name,
        {"sigma2",       "rho",          "sigma2_over_rho"},
        {BCParity::even, BCParity::even, BCParity::even   }
    };
    anisotropy_extraction.execute_query(&amr_ptr->anisotropy_interpolator,
                                        anisotropy_vars);
}

void EkpyroticLevel::print_diagnostics()
{
    const amrex::Real time = get_state_data(state_index).curTime();
    chi_relaxation_params_t relaxation;
    relaxation.fill_params();

    const auto constraints       = derive(Constraints::name, time, 0);
    const amrex::MultiFab &state = get_new_data(state_index);
    const auto num_points = static_cast<amrex::Real>(state.boxArray().numPts());

    const amrex::Real ham_rms   = constraints->norm2(0) / std::sqrt(num_points);
    const amrex::Real ham_max   = constraints->norm0(0);
    amrex::Real mom_sum_squares = 0.0;
    amrex::Real mom_max         = 0.0;
    for (int icomp = 1; icomp <= AMREX_SPACEDIM; ++icomp)
    {
        const amrex::Real norm2  = constraints->norm2(icomp);
        mom_sum_squares         += norm2 * norm2;
        mom_max                  = std::max(mom_max, constraints->norm0(icomp));
    }
    const amrex::Real mom_rms = std::sqrt(mom_sum_squares / num_points);

    amrex::Print() << std::setprecision(12) << std::scientific
                   << "[ekpyrotic] step " << get_gr_amr_ptr()->levelSteps(0)
                   << " t " << time << " relaxing "
                   << ((time < relaxation.time) ? 1 : 0) << " Ham_rms "
                   << ham_rms << " Ham_max " << ham_max << " Mom_rms "
                   << mom_rms << " Mom_max " << mom_max << " chi_mean "
                   << state.sum(c_chi) / num_points << " chi_min "
                   << state.min(c_chi) << " chi_max " << state.max(c_chi)
                   << " K_mean " << state.sum(c_K) / num_points
                   << " lapse_mean " << state.sum(c_lapse) / num_points << "\n"
                   << std::defaultfloat;
}

void EkpyroticLevel::initData()
{
    BL_PROFILE("EkpyroticLevel::initData()");

    if (get_gr_amr_ptr()->Verbose() > 0)
    {
        amrex::Print() << "EkpyroticLevel::initData " << Level() << "\n";
    }

    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    const EkpyroticInitialData initial_data(Geom().CellSize(0));
    static_assert(std::is_trivially_copyable_v<EkpyroticInitialData>,
                  "EkpyroticInitialData must be device copyable");

    // h_ij = delta_ij so Gamma^i = 0 and, with zero shift, B^i = 0: no need
    // for the GammaCalculator or set_initial_B_to_Gamma
    amrex::ParallelFor(state_new, state_new.nGrowVect(),
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       { initial_data(ix, iy, iz, state_arrays[box_no]); });

    amrex::Gpu::streamSynchronize();
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void EkpyroticLevel::specific_eval_rhs(amrex::MultiFab &a_soln,
                                       amrex::MultiFab &a_rhs,
                                       const amrex::Real /*a_time*/)
{
    BL_PROFILE("EkpyroticLevel::specific_eval_rhs()");

    const auto &soln_arrays = a_soln.arrays();

    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;
    const PositiveChiAndLapse positive_chi_and_lapse;

    amrex::ParallelFor(
        a_soln, a_soln.nGrowVect(),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]);
            positive_chi_and_lapse(ix, iy, iz, soln_arrays[box_no]);
        });

    // Decide per step (not per RK stage) whether this is a relaxation step so
    // that a step is never a mixture of relaxation and evolution stages. During
    // advance() the previous time of the state is the start of the step.
    chi_relaxation_params_t relaxation;
    relaxation.fill_params();
    const amrex::Real step_start_time = get_state_data(state_index).prevTime();
    if (step_start_time < relaxation.time)
    {
        relaxation_rhs(a_soln, a_rhs, relaxation.speed);
        return;
    }

    if (m_evolution_spatial_derivative_order == 4)
    {
        evolution_rhs<FourthOrderDerivatives>(a_soln, a_rhs);
    }
    else if (m_evolution_spatial_derivative_order == 6)
    {
        evolution_rhs<SixthOrderDerivatives>(a_soln, a_rhs);
    }
    else
    {
        amrex::Abort("spatial_derivative_order must be 4 or 6");
    }
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void EkpyroticLevel::relaxation_rhs(amrex::MultiFab &a_soln,
                                    amrex::MultiFab &a_rhs,
                                    const amrex::Real a_relaxation_speed)
{
    BL_PROFILE("EkpyroticLevel::relaxation_rhs()");

    const auto &soln_arrays       = a_soln.arrays();
    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays        = a_rhs.arrays();

    // Pi = chi^{3/2} Q and A_ij = chi^{3/2} A^Xue_ij from the current chi
    // (ghost cells included as they are used for the derivatives of A_ij)
    const EkpyroticInitialData initial_data(Geom().CellSize(0));
    amrex::ParallelFor(
        a_soln, a_soln.nGrowVect(),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        { initial_data.set_momenta(ix, iy, iz, soln_arrays[box_no]); });

    // Only chi is relaxed; write Ham directly into the chi component of the
    // RHS (no momentum constraints are computed)
    a_rhs.setVal(0.0);
    const EkpyroticConstraints constraints(Geom().CellSize(0), c_chi,
                                           Interval());
    amrex::ParallelFor(a_rhs,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           constraints(ix, iy, iz, rhs_arrays[box_no],
                                       const_soln_arrays[box_no]);
                           rhs_arrays[box_no](ix, iy, iz, c_chi) *=
                               a_relaxation_speed;
                       });

    amrex::Gpu::streamSynchronize();
}

template <class deriv_t>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void EkpyroticLevel::evolution_rhs(amrex::MultiFab &a_soln,
                                   amrex::MultiFab &a_rhs)
{
    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays        = a_rhs.arrays();

    const CCZ4RHSWithMatter<ScalarFieldWithPotential<deriv_t>, deriv_t>
        ccz4_rhs(Geom().CellSize(0));
    const IntegratedMovingPunctureGauge<deriv_t>
        integrated_moving_puncture_gauge(Geom().CellSize(0));

    amrex::ParallelFor(a_rhs,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           ccz4_rhs.compute_chi_and_h_ij(
                               ix, iy, iz, rhs_arrays[box_no],
                               const_soln_arrays[box_no]);
                       });

    // NOLINTBEGIN(bugprone-easily-swappable-parameters)
    amrex::ParallelFor(a_rhs,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           ccz4_rhs.compute_A_ij_and_Theta_and_Gamma(
                               ix, iy, iz, rhs_arrays[box_no],
                               const_soln_arrays[box_no]);
                       });
    // NOLINTEND(bugprone-easily-swappable-parameters)

    amrex::ParallelFor(
        a_rhs,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            ccz4_rhs.add_emtensor_rhs(ix, iy, iz, rhs_arrays[box_no],
                                      const_soln_arrays[box_no]);
            integrated_moving_puncture_gauge.calculate_rhs(
                ix, iy, iz, rhs_arrays[box_no], const_soln_arrays[box_no]);
            ccz4_rhs.add_matter_rhs(ix, iy, iz, rhs_arrays[box_no],
                                    const_soln_arrays[box_no]);
            ccz4_rhs.apply_dissipation(ix, iy, iz, rhs_arrays[box_no],
                                       const_soln_arrays[box_no]);
        });

    amrex::Gpu::streamSynchronize();
}

void EkpyroticLevel::specific_update_ode(amrex::MultiFab &a_soln)
{
    BL_PROFILE("EkpyroticLevel::specific_update_ode()");

    const auto &soln_arrays = a_soln.arrays();
    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;

    amrex::ParallelFor(
        a_soln, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        { algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]); });
    amrex::Gpu::streamSynchronize();
}

void EkpyroticLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array,
                               const amrex::Real /*a_regrid_threshold*/)
{
    BL_PROFILE("EkpyroticLevel::tag_cells()");

    const auto &tag_arrays = a_tag_box_array.arrays();

    const FixedGridsTagger tagger(Geom().CellSize(0), Level());

    amrex::ParallelFor(a_tag_box_array,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       { tagger(ix, iy, iz, tag_arrays[box_no]); });
    amrex::Gpu::streamSynchronize();
}
