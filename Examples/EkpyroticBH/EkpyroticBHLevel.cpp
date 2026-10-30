/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#include "EkpyroticBHLevel.hpp"

#include "AlgebraicConstraintsEnforcer.hpp"
#include "CCZ4RHSWithMatter.hpp"
#include "ConstraintsWithMatter.hpp"
#include "DiagnosticsParameters.hpp"
#include "EkpyroticBHData.hpp"
#include "FixedGridsTagger.hpp"
#include "FourthOrderDerivatives.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
#include "LineExtraction.hpp"
#include "PositiveChiAndLapse.hpp"
#include "SixthOrderDerivatives.hpp"
#include "StateTypes.hpp"

#include <cmath>
#include <iomanip>
#include <type_traits>

using EkpyroticBHConstraints =
    ConstraintsWithMatter<EkpyroticBHLevel::ScalarFieldWithPotential<>>;

EkpyroticBHAmr *EkpyroticBHLevel::get_ekpyrotic_bh_amr_ptr()
{
    return dynamic_cast<EkpyroticBHAmr *>(get_gr_amr_ptr());
}

void EkpyroticBHLevel::variableSetUp()
{
    BL_PROFILE("EkpyroticBHLevel::variableSetUp()");
    state_variable_set_up();
    EkpyroticBHConstraints::set_up(state_index);
}

XCTSRelaxation EkpyroticBHLevel::make_relaxation() const
{
    xcts_params_t xcts;
    xcts.fill_params();
    const EkpyroticBHData data(Geom().CellSize(0));
    return {data, Geom().CellSize(0), xcts.coefficients};
}

bool EkpyroticBHLevel::is_relaxation_step()
{
    // Decided per step (not per RK stage) so that a step is never a mixture
    // of relaxation and evolution stages. During advance() the previous time
    // of the state is the start of the step.
    xcts_params_t xcts;
    xcts.fill_params();
    return get_state_data(state_index).prevTime() < xcts.relaxation_time;
}

amrex::MultiFab EkpyroticBHLevel::filled_state()
{
    const amrex::MultiFab &state_new = get_new_data(state_index);
    const int num_ghosts             = state_new.nGrow();
    amrex::MultiFab filled(state_new.boxArray(), state_new.DistributionMap(),
                           NUM_VARS, num_ghosts);
    FillPatch(*this, filled, num_ghosts, get_state_data(state_index).curTime(),
              state_index, 0, NUM_VARS);
    return filled;
}

void EkpyroticBHLevel::update_bssn_variables()
{
    BL_PROFILE("EkpyroticBHLevel::update_bssn_variables()");

    // d_j beta^i is needed for Ahat, so use a copy with filled ghost cells
    const amrex::MultiFab filled = filled_state();
    const auto &filled_arrays    = filled.const_arrays();
    amrex::MultiFab &state_new   = get_new_data(state_index);
    const auto &state_arrays     = state_new.arrays();

    const XCTSRelaxation relaxation = make_relaxation();
    amrex::ParallelFor(state_new,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           relaxation.set_bssn_variables(ix, iy, iz,
                                                         state_arrays[box_no],
                                                         filled_arrays[box_no]);
                       });
    amrex::Gpu::streamSynchronize();
}

void EkpyroticBHLevel::hand_over_to_evolution(const xcts_params_t &a_xcts)
{
    const bool precollapse = (a_xcts.evolution_lapse == "precollapsed");
    const bool zero_shift  = (a_xcts.evolution_shift == "zero");
    if (Level() == 0)
    {
        amrex::Print() << "EkpyroticBHLevel: end of the XCTS relaxation, "
                       << "lapse = " << (precollapse ? "psi^-2" : "XCTS lapse")
                       << ", shift = " << (zero_shift ? "0" : "XCTS shift")
                       << "\n";
    }
    if (!precollapse && !zero_shift)
    {
        return;
    }

    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();
    amrex::ParallelFor(state_new,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           const auto &state = state_arrays[box_no];
                           if (precollapse)
                           {
                               state(ix, iy, iz, c_lapse) =
                                   std::sqrt(state(ix, iy, iz, c_chi));
                           }
                           if (zero_shift)
                           {
                               FOR (i)
                               {
                                   state(ix, iy, iz, c_shift1 + i)      = 0.0;
                                   state(ix, iy, iz, c_theta_beta1 + i) = 0.0;
                               }
                           }
                       });
    amrex::Gpu::streamSynchronize();
}

void EkpyroticBHLevel::specific_advance()
{
    BL_PROFILE("EkpyroticBHLevel::specific_advance()");

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
    amrex::Gpu::streamSynchronize();

    if (is_relaxation_step())
    {
        update_bssn_variables();

        xcts_params_t xcts;
        xcts.fill_params();
        if (get_state_data(state_index).curTime() >= xcts.relaxation_time)
        {
            hand_over_to_evolution(xcts);
        }
    }
}

void EkpyroticBHLevel::specific_post_init()
{
    BL_PROFILE("EkpyroticBHLevel::specific_post_init()");
    if (Level() == 0)
    {
        print_diagnostics();
    }
}

void EkpyroticBHLevel::specific_post_timestep()
{
    BL_PROFILE("EkpyroticBHLevel::specific_post_timestep()");

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
    EkpyroticBHAmr *amr_ptr = get_ekpyrotic_bh_amr_ptr();

    const LineExtraction<1> chi_extraction("chi_line_extraction", c_chi, dt,
                                           time, restart_time, first_step);
    chi_extraction.execute_query(&amr_ptr->chi_interpolator);

    const LineExtraction<1> lapse_extraction(
        "lapse_line_extraction", c_lapse, dt, time, restart_time, first_step);
    lapse_extraction.execute_query(&amr_ptr->lapse_interpolator);

    const LineExtraction<2> xcts_extraction("xcts_line_extraction", c_xcts_u,
                                            dt, time, restart_time, first_step);
    xcts_extraction.execute_query(&amr_ptr->xcts_interpolator);

    const LineExtraction<4> constraints_extraction(
        "constraints_line_extraction", 0, dt, time, restart_time, first_step);
    constexpr std::array<BCParity, 4> constraints_parities{
        BCParity::even, BCParity::odd_x, BCParity::odd_y, BCParity::odd_z};
    const LineExtraction<4>::derived_vars_t constraints_vars{
        .name            = Constraints::name,
        .component_names = {"Ham", "Mom1", "Mom2", "Mom3"},
        .parities        = constraints_parities
    };
    constraints_extraction.execute_query(&amr_ptr->constraints_interpolator,
                                         constraints_vars);
}

void EkpyroticBHLevel::print_diagnostics()
{
    // Called on level 0 once all finer levels have reached the same time
    for (int lev = 0; lev <= get_gr_amr_ptr()->finestLevel(); ++lev)
    {
        auto &level =
            dynamic_cast<EkpyroticBHLevel &>(get_gr_amr_ptr()->getLevel(lev));
        level.print_level_diagnostics();
    }
}

void EkpyroticBHLevel::print_level_diagnostics()
{
    const amrex::Real time = get_state_data(state_index).curTime();
    xcts_params_t xcts;
    xcts.fill_params();
    const amrex::Real excision_radius = xcts.diagnostic_excision_radius;

    const EkpyroticBHData data(Geom().CellSize(0));
    const XCTSRelaxation relaxation = make_relaxation();

    // XCTS residuals |R_u|, |R_v|, |R_beta| (zero inside excision_radius)
    const amrex::MultiFab filled = filled_state();
    const auto &filled_arrays    = filled.const_arrays();
    amrex::MultiFab residuals(filled.boxArray(), filled.DistributionMap(), 3,
                              0);
    const auto &residual_arrays = residuals.arrays();
    amrex::ParallelFor(
        residuals,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            const auto &out = residual_arrays[box_no];
            XCTSRelaxation::point_t point;
            const auto res =
                relaxation.residuals(ix, iy, iz, filled_arrays[box_no], point);
            const bool excised =
                EkpyroticBHData::radius(point.displacement) < excision_radius;
            amrex::Real beta2 = 0.0;
            FOR (i)
            {
                beta2 += res.beta(i) * res.beta(i);
            }
            out(ix, iy, iz, 0) = excised ? 0.0 : res.u;
            out(ix, iy, iz, 1) = excised ? 0.0 : res.v;
            out(ix, iy, iz, 2) = excised ? 0.0 : std::sqrt(beta2);
        });

    // BSSN constraints (zero inside excision_radius)
    const auto constraints        = derive(Constraints::name, time, 0);
    const auto &constraint_arrays = constraints->arrays();
    const int num_constraints     = constraints->nComp();
    amrex::ParallelFor(
        *constraints,
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        {
            const amrex::Real r =
                EkpyroticBHData::radius(data.displacement(ix, iy, iz));
            if (r < excision_radius)
            {
                for (int icomp = 0; icomp < num_constraints; ++icomp)
                {
                    constraint_arrays[box_no](ix, iy, iz, icomp) = 0.0;
                }
            }
        });
    amrex::Gpu::streamSynchronize();

    const amrex::MultiFab &state = get_new_data(state_index);
    const auto num_points = static_cast<amrex::Real>(state.boxArray().numPts());
    const amrex::Real sqrt_points = std::sqrt(num_points);

    const amrex::Real ham_rms   = constraints->norm2(0) / sqrt_points;
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

    amrex::Real beta_max = 0.0;
    for (int icomp = 0; icomp < AMREX_SPACEDIM; ++icomp)
    {
        beta_max = std::max(beta_max, state.norm0(c_shift1 + icomp));
    }

    amrex::Print() << std::setprecision(12) << std::scientific
                   << "[ekpyrotic_bh] step " << get_gr_amr_ptr()->levelSteps(0)
                   << " lev " << Level() << " t " << time << " relaxing "
                   << ((time < xcts.relaxation_time) ? 1 : 0) << " Ru_rms "
                   << residuals.norm2(0) / sqrt_points << " Ru_max "
                   << residuals.norm0(0) << " Rv_rms "
                   << residuals.norm2(1) / sqrt_points << " Rv_max "
                   << residuals.norm0(1) << " Rbeta_rms "
                   << residuals.norm2(2) / sqrt_points << " Rbeta_max "
                   << residuals.norm0(2) << " Ham_rms " << ham_rms
                   << " Ham_max " << ham_max << " Mom_rms " << mom_rms
                   << " Mom_max " << mom_max << " chi_min " << state.min(c_chi)
                   << " lapse_min " << state.min(c_lapse) << " lapse_max "
                   << state.max(c_lapse) << " u_mean "
                   << state.sum(c_xcts_u) / num_points << " beta_max "
                   << beta_max << "\n"
                   << std::defaultfloat;
}

void EkpyroticBHLevel::initData()
{
    BL_PROFILE("EkpyroticBHLevel::initData()");

    if (get_gr_amr_ptr()->Verbose() > 0)
    {
        amrex::Print() << "EkpyroticBHLevel::initData " << Level() << "\n";
    }

    amrex::MultiFab &state_new = get_new_data(state_index);
    const auto &state_arrays   = state_new.arrays();

    const XCTSRelaxation relaxation = make_relaxation();
    static_assert(std::is_trivially_copyable_v<XCTSRelaxation>,
                  "XCTSRelaxation must be device copyable");

    // u = v = 0, beta = 0 and theta = 0: the BSSN variables follow from the
    // puncture part and the free data alone
    amrex::ParallelFor(state_new, state_new.nGrowVect(),
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           const auto &state = state_arrays[box_no];
                           for (int icomp = 0; icomp < NUM_VARS; ++icomp)
                           {
                               state(ix, iy, iz, icomp) = 0.0;
                           }
                           relaxation.set_bssn_variables_zero_shift(ix, iy, iz,
                                                                    state);
                       });

    amrex::Gpu::streamSynchronize();
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void EkpyroticBHLevel::specific_eval_rhs(amrex::MultiFab &a_soln,
                                         amrex::MultiFab &a_rhs,
                                         const amrex::Real /*a_time*/)
{
    BL_PROFILE("EkpyroticBHLevel::specific_eval_rhs()");

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

    if (is_relaxation_step())
    {
        relaxation_rhs(a_soln, a_rhs);
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
void EkpyroticBHLevel::relaxation_rhs(amrex::MultiFab &a_soln,
                                      amrex::MultiFab &a_rhs)
{
    BL_PROFILE("EkpyroticBHLevel::relaxation_rhs()");

    const auto &const_soln_arrays = a_soln.const_arrays();
    const auto &rhs_arrays        = a_rhs.arrays();

    const XCTSRelaxation relaxation = make_relaxation();

    // All RHS apart from those of the XCTS unknowns are zero
    a_rhs.setVal(0.0);
    amrex::ParallelFor(a_rhs,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       {
                           relaxation.compute(ix, iy, iz, rhs_arrays[box_no],
                                              const_soln_arrays[box_no]);
                       });

    amrex::Gpu::streamSynchronize();
}

template <class deriv_t>
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void EkpyroticBHLevel::evolution_rhs(amrex::MultiFab &a_soln,
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
            // The XCTS variables are only used by the relaxation: freeze
            // them. Their RHS is not written above (and the dissipation adds
            // to the RHS of all NUM_VARS components), so overwrite it last
            for (int icomp = c_xcts_u; icomp <= c_theta_beta3; ++icomp)
            {
                rhs_arrays[box_no](ix, iy, iz, icomp) = 0.0;
            }
        });

    amrex::Gpu::streamSynchronize();
}

void EkpyroticBHLevel::specific_update_ode(amrex::MultiFab &a_soln)
{
    BL_PROFILE("EkpyroticBHLevel::specific_update_ode()");

    const auto &soln_arrays = a_soln.arrays();
    const AlgebraicConstraintsEnforcer algebraic_constraints_enforcer;

    amrex::ParallelFor(
        a_soln, amrex::IntVect(0),
        [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
        { algebraic_constraints_enforcer(ix, iy, iz, soln_arrays[box_no]); });
    amrex::Gpu::streamSynchronize();
}

void EkpyroticBHLevel::tag_cells(amrex::TagBoxArray &a_tag_box_array,
                                 const amrex::Real /*a_regrid_threshold*/)
{
    BL_PROFILE("EkpyroticBHLevel::tag_cells()");

    const auto &tag_arrays = a_tag_box_array.arrays();

    const FixedGridsTagger tagger(Geom().CellSize(0), Level());

    amrex::ParallelFor(a_tag_box_array,
                       [=] AMREX_GPU_DEVICE(int box_no, int ix, int iy, int iz)
                       { tagger(ix, iy, iz, tag_arrays[box_no]); });
    amrex::Gpu::streamSynchronize();
}
