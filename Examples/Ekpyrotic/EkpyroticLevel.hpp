/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef EKPYROTICLEVEL_HPP_
#define EKPYROTICLEVEL_HPP_

#include "DefaultLevelBld.hpp"
#include "EkpyroticAmr.hpp"
#include "FourthOrderDerivatives.hpp"
#include "GRAmrLevel.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"
#include "SixthOrderDerivatives.hpp"

/// Evolution level for a scalar field cosmology with an exponential
/// (ekpyrotic) potential and inhomogeneities along x.
/*!
    For time < relaxation.time the conformal factor chi is relaxed towards a
    solution of the Hamiltonian constraint: all RHS are zero apart from
    rhs_chi = relaxation.speed * Ham, and Pi and A_ij are recomputed from chi
    before every RK stage and after every step so that the momentum
    constraint holds exactly (see EkpyroticInitialData). After that the
    CCZ4/BSSN equations with the scalar field are evolved.
*/
class EkpyroticLevel : public GRAmrLevel
{
  public:
    using GRAmrLevel::GRAmrLevel;

    template <class deriv_t = FourthOrderDerivatives>
    using ScalarFieldWithPotential = ScalarField<Potential, deriv_t>;

    EkpyroticAmr *get_ekpyrotic_amr_ptr();

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
    //! RHS during the chi relaxation
    void relaxation_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs,
                        amrex::Real a_relaxation_speed);

    //! Full CCZ4/BSSN + scalar field RHS
    template <class deriv_t>
    void evolution_rhs(amrex::MultiFab &a_soln, amrex::MultiFab &a_rhs);

    //! Print constraint norms and averages of chi, K and the lapse
    void print_diagnostics();
};

#endif /* EKPYROTICLEVEL_HPP_ */
