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
    and their pseudo-time derivatives. The XCTS shift is stored directly in
    c_shift1-3 and its pseudo-time derivative in c_theta_beta1-3. All of these
    auxiliary variables are frozen after the relaxation.
*/
enum
{
    c_phi = NUM_CCZ4_VARS,
    c_Pi,

    c_xcts_u, // regular part of psi
    c_xcts_v, // regular part of alpha psi
    c_theta_u,
    c_theta_v,
    c_theta_beta1,
    c_theta_beta2,
    c_theta_beta3,

    NUM_VARS
};

namespace StateVariables
{
static const amrex::Vector<std::string> scalar_field_names = {
    "phi",     "Pi",          "xcts_u",      "xcts_v",     "theta_u",
    "theta_v", "theta_beta1", "theta_beta2", "theta_beta3"};

static const amrex::Vector<std::string> names =
    ArrayTools::concatenate(CCZ4StateVariables::names, scalar_field_names);

static constexpr int NUM_MATTER_VARS = NUM_VARS - NUM_CCZ4_VARS;

static const std::array<BCParity, NUM_MATTER_VARS> scalar_field_parities = {
    BCParity::even,  BCParity::even,  BCParity::even,
    BCParity::even,  BCParity::even,  BCParity::even,
    BCParity::odd_x, BCParity::odd_y, BCParity::odd_z};

static const std::array<BCParity, NUM_VARS> parities = ArrayTools::concatenate(
    CCZ4StateVariables::parities, scalar_field_parities);

static const std::array<amrex::Real, NUM_MATTER_VARS>
    scalar_field_asymptotic_values = {0.0, 0.0, 0.0, 0.0, 0.0,
                                      0.0, 0.0, 0.0, 0.0};

static const std::array<amrex::Real, NUM_VARS> asymptotic_values =
    ArrayTools::concatenate(CCZ4StateVariables::asymptotic_values,
                            scalar_field_asymptotic_values);
} // namespace StateVariables

#endif /* STATEVARIABLES_HPP */
