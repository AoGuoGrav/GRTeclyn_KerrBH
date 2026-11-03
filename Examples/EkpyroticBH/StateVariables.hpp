/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef STATEVARIABLES_HPP
#define STATEVARIABLES_HPP

#include "ArrayTools.hpp"
#include "BCParity.hpp"
#include "CCZ4StateVariables.hpp"

/// This enum gives the index of every variable stored in the grid.
/*!
    Besides the scalar field, the grid carries the unknowns of the XCTS
    relaxation (see XCTSRelaxation): psi = 1 + f_BH + xcts_u and
    alpha psi = 1 + f_BH + xcts_v, where f_BH is the (analytic) puncture part,
    and the pseudo-time derivative theta_v of xcts_v. The XCTS shift is stored
    directly in c_shift1-3. xcts_u, xcts_v and theta_v are frozen after the
    relaxation.
*/
enum
{
    c_phi = NUM_CCZ4_VARS,
    c_Pi,

    c_xcts_u, // regular part of psi
    c_xcts_v, // regular part of alpha psi
    c_theta_v,

    NUM_VARS
};

/// Pseudo-time derivatives of xcts_u and beta^i during the relaxation
/*!
    They are stored in CCZ4 variables that are constant during the relaxation
    and are not read by the relaxation or the constraint diagnostics: Theta
    (zero for the XCTS data) and B^i (the Gamma-driver auxiliary variable,
    with the same parity as beta^i). Both are set to zero at the end of the
    relaxation (see EkpyroticBHLevel::hand_over_to_evolution).
*/
enum
{
    c_theta_u     = c_Theta,
    c_theta_beta1 = c_B1,
    c_theta_beta2 = c_B2,
    c_theta_beta3 = c_B3
};

namespace StateVariables
{
static const amrex::Vector<std::string> scalar_field_names = {
    "phi", "Pi", "xcts_u", "xcts_v", "theta_v"};

static const amrex::Vector<std::string> names =
    ArrayTools::concatenate(CCZ4StateVariables::names, scalar_field_names);

static constexpr int NUM_MATTER_VARS = NUM_VARS - NUM_CCZ4_VARS;

static const std::array<BCParity, NUM_MATTER_VARS> scalar_field_parities = {
    BCParity::even, BCParity::even, BCParity::even, BCParity::even,
    BCParity::even};

static const std::array<BCParity, NUM_VARS> parities = ArrayTools::concatenate(
    CCZ4StateVariables::parities, scalar_field_parities);

static const std::array<amrex::Real, NUM_MATTER_VARS>
    scalar_field_asymptotic_values = {0.0, 0.0, 0.0, 0.0, 0.0};

static const std::array<amrex::Real, NUM_VARS> asymptotic_values =
    ArrayTools::concatenate(CCZ4StateVariables::asymptotic_values,
                            scalar_field_asymptotic_values);
} // namespace StateVariables

#endif /* STATEVARIABLES_HPP */
