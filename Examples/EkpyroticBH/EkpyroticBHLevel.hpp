/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef EKPYROTICBHLEVEL_HPP_
#define EKPYROTICBHLEVEL_HPP_

#include "DefaultLevelBld.hpp"
#include "EkpyroticBHAmr.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRAmrLevel.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"
#include "SixthOrderDerivatives.hpp"
#include "XCTSParameters.hpp"
#include "XCTSRelaxation.hpp"

/// Evolution level for a black hole in a scalar field cosmology with an
/// exponential (ekpyrotic) potential
/*!
    For time < xcts.relaxation_time the XCTS unknowns (u, v, beta^i, see
    XCTSRelaxation) are relaxed and all other RHS are zero; the BSSN
    variables are recomputed from the XCTS unknowns after every step. At the
    end of the relaxation the lapse and shift are optionally replaced (see
    xcts_params_t). After that the CCZ4/BSSN equations with the scalar field
    are evolved, with the XCTS auxiliary variables frozen.
*/
class EkpyroticBHLevel : public GRAmrLevel
{
  public:
    using GRAmrLevel::GRAmrLevel;

    template <class deriv_t = FourthOrderDerivatives>
    using ScalarFieldWithPotential = ScalarField<Potential, deriv_t>;

    EkpyroticBHAmr *get_ekpyrotic_bh_amr_ptr();

    static void variableSetUp();

    void specific_advance() override;

    void initData() override;

    void specific_eval_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs,
                           amrex::Real a_time) override;

    void specific_update_ode(amrex::MultiFab &a_soln) override;

    void specific_post_init() override;

    void specific_post_timestep() override;

    void tag_cells(amrex::TagBoxArray &a_tag_box_array,
                   amrex::Real a_regrid_threshold) final;

  protected:
    //! The relaxation operator for this level
    [[nodiscard]] XCTSRelaxation make_relaxation() const;

    //! Whether the step that starts at prevTime is a relaxation step
    [[nodiscard]] bool is_relaxation_step();

    //! RHS during the XCTS relaxation
    void relaxation_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs);

    //! Full CCZ4/BSSN + scalar field RHS
    template <class deriv_t>
    void evolution_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs);

    //! A copy of the new state with filled ghost cells
    amrex::MultiFab filled_state();

    //! Recompute the BSSN variables from the XCTS unknowns
    void update_bssn_variables();

    //! Replace the lapse/shift at the end of the relaxation (if requested)
    void hand_over_to_evolution(const xcts_params_t &a_xcts);

    //! Print the diagnostics of every level (call on level 0)
    void print_diagnostics();

    //! Print XCTS residuals, constraint norms and averages on this level
    void print_level_diagnostics();
};

#endif /* EKPYROTICBHLEVEL_HPP_ */
