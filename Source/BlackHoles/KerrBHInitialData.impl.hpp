/* GRTeclyn
 *
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#if !defined(KERRBHINITIALDATA_HPP_)
#error "This file should only be included through KerrBHInitialData.hpp"
#endif

#ifndef KERRBHINITIALDATA_IMPL_HPP_
#define KERRBHINITIALDATA_IMPL_HPP_

#include "GRParmParse.hpp"

#include <cmath>

namespace
{
// Analytic single-Kerr initial data is built in a local frame,
// then converted to the global Cartesian frame and stored as CCZ4/BSSN
// variables.
using kerr_vector_t = std::array<amrex::Real, AMREX_SPACEDIM>;
using kerr_matrix_t = std::array<kerr_vector_t, AMREX_SPACEDIM>;

// Return the value unless it is smaller than the requested numerical floor.
// This keeps square roots and divisions away from invalid values on the GPU.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real kerr_max(const amrex::Real a_value, const amrex::Real a_minimum)
{
    return (a_value > a_minimum) ? a_value : a_minimum;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real kerr_abs(const amrex::Real a_value)
{
    return (a_value >= 0.0) ? a_value : -a_value;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real dot_product(const kerr_vector_t &a_left,
                         const kerr_vector_t &a_right)
{
    amrex::Real result = 0.0;

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
    {
        result += a_left[idir] * a_right[idir];
    }

    return result;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
kerr_vector_t cross_product(const kerr_vector_t &a_left,
                            const kerr_vector_t &a_right)
{
    return {a_left[1] * a_right[2] - a_left[2] * a_right[1],
            a_left[2] * a_right[0] - a_left[0] * a_right[2],
            a_left[0] * a_right[1] - a_left[1] * a_right[0]};
}

// Normalize the requested spin direction to obtain the local Kerr axis.
// A zero vector falls back to the standard +z direction.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
kerr_vector_t normalise(const kerr_vector_t &a_vector)
{
    const amrex::Real norm =
        std::sqrt(kerr_max(dot_product(a_vector, a_vector), 0.0));

    if (norm <= 0.0)
    {
        return {0.0, 0.0, 1.0};
    }

    return {a_vector[0] / norm, a_vector[1] / norm,
            a_vector[2] / norm};
}

// The returned matrix R maps global Cartesian components to local components:
//
//     x_local^a = R^a_i x_global^i
//
// The local z axis is aligned with the requested physical spin direction.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
kerr_matrix_t rotation_global_to_local(
    const kerr_vector_t &a_spin_direction)
{
    const kerr_vector_t local_z = normalise(a_spin_direction);

    kerr_vector_t reference_axis{};

    // This choice makes R the identity matrix when spin_direction = (0, 0, 1).
    if (kerr_abs(local_z[2]) > 0.9)
    {
        reference_axis = {0.0, 1.0, 0.0};
    }
    else
    {
        reference_axis = {0.0, 0.0, 1.0};
    }

    const kerr_vector_t local_x =
        normalise(cross_product(reference_axis, local_z));

    const kerr_vector_t local_y = cross_product(local_z, local_x);

    return {local_x, local_y, local_z};
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
kerr_vector_t rotate_global_vector_to_local(
    const kerr_matrix_t &a_rotation, const kerr_vector_t &a_global_vector)
{
    kerr_vector_t local_vector{};

    for (int ilocal = 0; ilocal < AMREX_SPACEDIM; ++ilocal)
    {
        for (int iglobal = 0; iglobal < AMREX_SPACEDIM; ++iglobal)
        {
            local_vector[ilocal] +=
                a_rotation[ilocal][iglobal] * a_global_vector[iglobal];
        }
    }

    return local_vector;
}

// For x_local = R x_global, a local contravariant vector transforms as
//
//     v_global = R^T v_local.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
kerr_vector_t rotate_local_vector_to_global(
    const kerr_matrix_t &a_rotation, const kerr_vector_t &a_local_vector)
{
    kerr_vector_t global_vector{};

    for (int iglobal = 0; iglobal < AMREX_SPACEDIM; ++iglobal)
    {
        for (int ilocal = 0; ilocal < AMREX_SPACEDIM; ++ilocal)
        {
            global_vector[iglobal] +=
                a_rotation[ilocal][iglobal] * a_local_vector[ilocal];
        }
    }

    return global_vector;
}

// For x_local = R x_global, a covariant rank-two tensor transforms as
//
//     T_global = R^T T_local R.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
kerr_matrix_t rotate_local_tensor_to_global(
    const kerr_matrix_t &a_rotation, const kerr_matrix_t &a_local_tensor)
{
    kerr_matrix_t global_tensor{};

    for (int iglobal_i = 0; iglobal_i < AMREX_SPACEDIM; ++iglobal_i)
    {
        for (int iglobal_j = 0; iglobal_j < AMREX_SPACEDIM; ++iglobal_j)
        {
            for (int ilocal_i = 0; ilocal_i < AMREX_SPACEDIM; ++ilocal_i)
            {
                for (int ilocal_j = 0; ilocal_j < AMREX_SPACEDIM; ++ilocal_j)
                {
                    global_tensor[iglobal_i][iglobal_j] +=
                        a_rotation[ilocal_i][iglobal_i] *
                        a_local_tensor[ilocal_i][ilocal_j] *
                        a_rotation[ilocal_j][iglobal_j];
                }
            }
        }
    }

    return global_tensor;
}

// Convert covariant tensors and a contravariant vector from
// (r, theta, phi) to local Cartesian coordinates.
// The Jacobians below implement the tensor transformation rules.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
void spherical_to_cartesian(
    const kerr_matrix_t &a_spherical_metric,
    const kerr_matrix_t &a_spherical_extrinsic_curvature,
    const kerr_vector_t &a_spherical_shift,
    const kerr_vector_t &a_local_coordinates,
    kerr_matrix_t &a_cartesian_metric,
    kerr_matrix_t &a_cartesian_extrinsic_curvature,
    kerr_vector_t &a_cartesian_shift)
{
    const amrex::Real x = a_local_coordinates[0];
    const amrex::Real y = a_local_coordinates[1];
    const amrex::Real z = a_local_coordinates[2];

    constexpr amrex::Real coordinate_epsilon = 1.0e-12;

    const amrex::Real radius_squared = x * x + y * y + z * z;
    const amrex::Real radius =
        std::sqrt(kerr_max(radius_squared, coordinate_epsilon));

    const amrex::Real cylindrical_radius_squared =
        kerr_max(x * x + y * y, coordinate_epsilon);
    const amrex::Real cylindrical_radius =
        std::sqrt(cylindrical_radius_squared);

    // q_derivatives[a][i] = d q^a / d x^i, with q^a = (r, theta, phi).
    kerr_matrix_t q_derivatives{};

    q_derivatives[0][0] = x / radius;
    q_derivatives[0][1] = y / radius;
    q_derivatives[0][2] = z / radius;

    q_derivatives[1][0] =
        x * z / (radius_squared * cylindrical_radius);
    q_derivatives[1][1] =
        y * z / (radius_squared * cylindrical_radius);
    q_derivatives[1][2] = -cylindrical_radius / radius_squared;

    q_derivatives[2][0] = -y / cylindrical_radius_squared;
    q_derivatives[2][1] = x / cylindrical_radius_squared;
    q_derivatives[2][2] = 0.0;

    // Cartesian Jacobian J[i][a] = d x^i / d q^a.
    kerr_matrix_t cartesian_jacobian{};

    cartesian_jacobian[0][0] = x / radius;
    cartesian_jacobian[1][0] = y / radius;
    cartesian_jacobian[2][0] = z / radius;

    cartesian_jacobian[0][1] = z * x / cylindrical_radius;
    cartesian_jacobian[1][1] = z * y / cylindrical_radius;
    cartesian_jacobian[2][1] = -cylindrical_radius;

    cartesian_jacobian[0][2] = -y;
    cartesian_jacobian[1][2] = x;
    cartesian_jacobian[2][2] = 0.0;

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        for (int a = 0; a < AMREX_SPACEDIM; ++a)
        {
            a_cartesian_shift[i] +=
                cartesian_jacobian[i][a] * a_spherical_shift[a];
        }

        for (int j = 0; j < AMREX_SPACEDIM; ++j)
        {
            for (int a = 0; a < AMREX_SPACEDIM; ++a)
            {
                for (int b = 0; b < AMREX_SPACEDIM; ++b)
                {
                    a_cartesian_metric[i][j] +=
                        a_spherical_metric[a][b] *
                        q_derivatives[a][i] * q_derivatives[b][j];

                    a_cartesian_extrinsic_curvature[i][j] +=
                        a_spherical_extrinsic_curvature[a][b] *
                        q_derivatives[a][i] * q_derivatives[b][j];
                }
            }
        }
    }
}

// The determinant of the physical spatial metric is used to construct
// the BSSN/CCZ4 conformal factor chi.
AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real determinant(const kerr_matrix_t &a_matrix)
{
    return a_matrix[0][0] *
               (a_matrix[1][1] * a_matrix[2][2] -
                a_matrix[1][2] * a_matrix[2][1]) -
           a_matrix[0][1] *
               (a_matrix[1][0] * a_matrix[2][2] -
                a_matrix[1][2] * a_matrix[2][0]) +
           a_matrix[0][2] *
               (a_matrix[1][0] * a_matrix[2][1] -
                a_matrix[1][1] * a_matrix[2][0]);
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
kerr_matrix_t inverse(const kerr_matrix_t &a_matrix,
                      const amrex::Real a_determinant)
{
    kerr_matrix_t inverse_matrix{};

    inverse_matrix[0][0] =
        (a_matrix[1][1] * a_matrix[2][2] -
         a_matrix[1][2] * a_matrix[2][1]) /
        a_determinant;

    inverse_matrix[0][1] =
        (a_matrix[0][2] * a_matrix[2][1] -
         a_matrix[0][1] * a_matrix[2][2]) /
        a_determinant;

    inverse_matrix[0][2] =
        (a_matrix[0][1] * a_matrix[1][2] -
         a_matrix[0][2] * a_matrix[1][1]) /
        a_determinant;

    inverse_matrix[1][0] =
        (a_matrix[1][2] * a_matrix[2][0] -
         a_matrix[1][0] * a_matrix[2][2]) /
        a_determinant;

    inverse_matrix[1][1] =
        (a_matrix[0][0] * a_matrix[2][2] -
         a_matrix[0][2] * a_matrix[2][0]) /
        a_determinant;

    inverse_matrix[1][2] =
        (a_matrix[0][2] * a_matrix[1][0] -
         a_matrix[0][0] * a_matrix[1][2]) /
        a_determinant;

    inverse_matrix[2][0] =
        (a_matrix[1][0] * a_matrix[2][1] -
         a_matrix[1][1] * a_matrix[2][0]) /
        a_determinant;

    inverse_matrix[2][1] =
        (a_matrix[0][1] * a_matrix[2][0] -
         a_matrix[0][0] * a_matrix[2][1]) /
        a_determinant;

    inverse_matrix[2][2] =
        (a_matrix[0][0] * a_matrix[1][1] -
         a_matrix[0][1] * a_matrix[1][0]) /
        a_determinant;

    return inverse_matrix;
}

AMREX_GPU_DEVICE AMREX_FORCE_INLINE
amrex::Real trace(const kerr_matrix_t &a_covariant_tensor,
                  const kerr_matrix_t &a_inverse_metric)
{
    amrex::Real result = 0.0;

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        for (int j = 0; j < AMREX_SPACEDIM; ++j)
        {
            result += a_inverse_metric[i][j] * a_covariant_tensor[i][j];
        }
    }

    return result;
}
} // namespace

<<<<<<< Updated upstream
void KerrBHInitialData::params_t::check_params()
=======
// Validate the physical Kerr parameters before they are used on the GPU.
inline void KerrBHInitialData::params_t::check_params()
>>>>>>> Stashed changes
{
    GRParmParse kerr_pp("kerr");

    amrex::Real checked_mass{};
    amrex::Real checked_spin{};

    kerr_pp.get("mass", checked_mass);
    kerr_pp.get("spin", checked_spin);

    if (checked_mass <= 0.0)
    {
        kerr_pp.error("mass", "must be > 0");
    }

    // spin is the Kerr parameter a = J / M.
    if (std::abs(checked_spin) > checked_mass)
    {
        kerr_pp.error("spin", "must satisfy abs(spin) <= mass");
    }

    std::array<amrex::Real, AMREX_SPACEDIM> checked_spin_direction{
        0.0, 0.0, 1.0};

    kerr_pp.queryAdd("spin_direction", checked_spin_direction);

    const amrex::Real spin_direction_norm_squared =
        checked_spin_direction[0] * checked_spin_direction[0] +
        checked_spin_direction[1] * checked_spin_direction[1] +
        checked_spin_direction[2] * checked_spin_direction[2];

    if (spin_direction_norm_squared <= 0.0)
    {
        kerr_pp.error("spin_direction", "must not be the zero vector");
    }
}

<<<<<<< Updated upstream
void KerrBHInitialData::params_t::fill_params()
=======
// Read the Kerr parameters and use geometry.center as the default black-hole centre.
inline void KerrBHInitialData::params_t::fill_params()
>>>>>>> Stashed changes
{
    GRParmParse kerr_pp("kerr");
    GRParmParse geometry_pp("geometry");

    kerr_pp.get("mass", mass);
    kerr_pp.get("spin", spin);

    // geometry.center is the default black-hole centre.
    geometry_pp.get("center", center);

    // If present, kerr.center overrides geometry.center.
    kerr_pp.queryAdd("center", center);

    std::array<amrex::Real, AMREX_SPACEDIM> offset{};
    kerr_pp.queryAdd("offset", offset);

    for (int idir = 0; idir < AMREX_SPACEDIM; ++idir)
    {
        center[idir] += offset[idir];
    }

    kerr_pp.queryAdd("spin_direction", spin_direction);
}

// Construct the initial-data object and load the validated parameter values.
AMREX_FORCE_INLINE
KerrBHInitialData::KerrBHInitialData(amrex::Real a_dx) : m_dx(a_dx)
{
    m_params.check_params();
    m_params.fill_params();
}

AMREX_FORCE_INLINE
// Evaluate the analytic Kerr initial data at one cell and write all state variables.
AMREX_GPU_DEVICE void KerrBHInitialData::operator()(
    int ix, int iy, int iz,
    const amrex::Array4<amrex::Real> &state) const
{
    amrex::CellData<amrex::Real> cell = state.cellData(ix, iy, iz);

    // Coordinates are measured relative to the configured black-hole centre.
    const Coordinates global_coordinates(
        amrex::IntVect(ix, iy, iz), m_dx, m_params.center);

    const vector_t global_xyz{global_coordinates.x, global_coordinates.y,
                              global_coordinates.z};

    const vector_t requested_spin_direction{
        m_params.spin_direction[0], m_params.spin_direction[1],
        m_params.spin_direction[2]};

    // The analytic Kerr formula is evaluated in a local frame where the
    // requested spin direction becomes the local z axis.
    const matrix_t rotation =
        rotation_global_to_local(requested_spin_direction);

    const vector_t local_xyz =
        rotate_global_vector_to_local(rotation, global_xyz);

    matrix_t spherical_metric{};
    matrix_t spherical_extrinsic_curvature{};
    vector_t spherical_shift{};
    amrex::Real analytic_lapse{};

    compute_kerr(spherical_metric, spherical_extrinsic_curvature,
                 spherical_shift, analytic_lapse, local_xyz);

    matrix_t local_cartesian_metric{};
    matrix_t local_cartesian_extrinsic_curvature{};
    vector_t local_cartesian_shift{};

    spherical_to_cartesian(
        spherical_metric, spherical_extrinsic_curvature, spherical_shift,
        local_xyz, local_cartesian_metric,
        local_cartesian_extrinsic_curvature, local_cartesian_shift);

    // Rotate the local analytic tensors back to the global Cartesian frame.
    const matrix_t physical_metric =
        rotate_local_tensor_to_global(rotation, local_cartesian_metric);

    const matrix_t physical_extrinsic_curvature =
        rotate_local_tensor_to_global(
            rotation, local_cartesian_extrinsic_curvature);

    const vector_t physical_shift =
        rotate_local_vector_to_global(rotation, local_cartesian_shift);

    // In three spatial dimensions, chi = det(gamma_ij)^(-1/3).
    // With h_ij = chi * gamma_ij, the conformal metric has unit determinant.
    const amrex::Real metric_determinant = determinant(physical_metric);

    const matrix_t inverse_physical_metric =
        inverse(physical_metric, metric_determinant);

    const amrex::Real chi = std::pow(metric_determinant, -1.0 / 3.0);

    const amrex::Real trace_K =
        trace(physical_extrinsic_curvature, inverse_physical_metric);

    matrix_t physical_trace_free_extrinsic_curvature{};

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        for (int j = 0; j < AMREX_SPACEDIM; ++j)
        {
            physical_trace_free_extrinsic_curvature[i][j] =
                physical_extrinsic_curvature[i][j] -
                physical_metric[i][j] * trace_K / 3.0;
        }
    }

    // Store the conformal factor used by the CCZ4/BSSN evolution variables.
    cell[c_chi] = chi;
    cell[c_K] = trace_K;

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        for (int j = i; j < AMREX_SPACEDIM; ++j)
        {
            cell[sym_var_idx(c_h11, i, j)] =
                chi * physical_metric[i][j];

            cell[sym_var_idx(c_A11, i, j)] =
                chi * physical_trace_free_extrinsic_curvature[i][j];
        }
    }

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        cell[c_shift1 + i] = physical_shift[i];
    }

    // Use the pre-collapsed moving-puncture lapse choice alpha = sqrt(chi).
    // The analytic Kerr lapse computed below is intentionally not stored here.
    // Use the pre-collapsed moving-puncture lapse choice alpha = sqrt(chi).
    // The analytic Kerr lapse computed below is intentionally not stored here.
    cell[c_lapse] = std::sqrt(chi);

    // Theta, Gamma^i and B^i remain zero here. KerrBHLevel::initData()
    // subsequently computes Gamma^i and initialises B^i = Gamma^i.
    static_cast<void>(analytic_lapse);
}

AMREX_GPU_DEVICE
void KerrBHInitialData::compute_kerr(
    matrix_t &spherical_g, matrix_t &spherical_K,
    vector_t &spherical_shift, amrex::Real &kerr_lapse,
    const vector_t &coords) const
{
    const amrex::Real mass = m_params.mass;
    const amrex::Real spin = m_params.spin;

    const amrex::Real x = coords[0];
    const amrex::Real y = coords[1];
    const amrex::Real z = coords[2];

    constexpr amrex::Real coordinate_epsilon = 1.0e-12;

    const amrex::Real raw_radius =
        std::sqrt(x * x + y * y + z * z);

    // The grid normally does not contain r = 0, but this prevents divisions
    // by zero if a cell happens to coincide with the puncture.
    const amrex::Real radius = kerr_max(raw_radius, 1.0e-6);

    const amrex::Real radius_squared = radius * radius;

    const amrex::Real cylindrical_radius_squared =
        kerr_max(x * x + y * y, coordinate_epsilon);

    const amrex::Real cylindrical_radius =
        std::sqrt(cylindrical_radius_squared);

    const amrex::Real cos_theta = z / radius;
    const amrex::Real sin_theta = cylindrical_radius / radius;

    const amrex::Real cos_theta_squared = cos_theta * cos_theta;
    const amrex::Real sin_theta_squared = sin_theta * sin_theta;

    const amrex::Real spin_squared = spin * spin;

    const amrex::Real horizon_root =
        std::sqrt(kerr_max(mass * mass - spin_squared, 0.0));

    const amrex::Real r_plus = mass + horizon_root;
    const amrex::Real r_minus = mass - horizon_root;

    // Convert the semi-isotropic radius used by the initial-data chart
    // to the Boyer-Lindquist radial coordinate used in the analytic formulas.
    const amrex::Real radial_factor =
        1.0 + 0.25 * r_plus / radius;

    const amrex::Real r_bl = radius * radial_factor * radial_factor;
    const amrex::Real r_bl_squared = r_bl * r_bl;

    const amrex::Real sigma =
        r_bl_squared + spin_squared * cos_theta_squared;

    const amrex::Real delta =
        r_bl_squared - 2.0 * mass * r_bl + spin_squared;

    // Named A in the analytic Kerr expressions.
    // Kerr auxiliary quantity A used by the angular metric and shift.
    const amrex::Real kerr_A =
        (r_bl_squared + spin_squared) *
            (r_bl_squared + spin_squared) -
        delta * spin_squared * sin_theta_squared;

    const amrex::Real gamma_rr =
        sigma * (radius + 0.25 * r_plus) *
            (radius + 0.25 * r_plus) /
        (radius * radius_squared * (r_bl - r_minus));

    for (int i = 0; i < AMREX_SPACEDIM; ++i)
    {
        for (int j = 0; j < AMREX_SPACEDIM; ++j)
        {
            spherical_g[i][j] = 0.0;
            spherical_K[i][j] = 0.0;
        }

        spherical_shift[i] = 0.0;
    }

    spherical_g[0][0] = gamma_rr;
    spherical_g[1][1] = sigma;
    spherical_g[2][2] = kerr_A * sin_theta_squared / sigma;

    const amrex::Real square_root_A_sigma =
        std::sqrt(kerr_A * sigma);

    const amrex::Real r_phi_denominator =
        sigma * square_root_A_sigma;

    const amrex::Real r_bl_fourth_power =
        r_bl_squared * r_bl_squared;

    const amrex::Real spin_fourth_power =
        spin_squared * spin_squared;

    // Nonzero components of the analytic Kerr extrinsic curvature in
    // spherical coordinates; symmetric partners are filled below.
    spherical_K[0][2] =
        spin * mass * sin_theta_squared / r_phi_denominator *
        (3.0 * r_bl_fourth_power +
         2.0 * spin_squared * r_bl_squared -
         spin_fourth_power -
         spin_squared * (r_bl_squared - spin_squared) *
             sin_theta_squared) *
        radial_factor /
        std::sqrt(radius * (r_bl - r_minus));

    spherical_K[2][0] = spherical_K[0][2];

    spherical_K[1][2] =
        -2.0 * spin * spin_squared * mass * r_bl * cos_theta *
        sin_theta * sin_theta_squared / r_phi_denominator *
        (radius - 0.25 * r_plus) *
        std::sqrt((r_bl - r_minus) / radius);

    spherical_K[2][1] = spherical_K[1][2];

    // Analytic Kerr lapse in the semi-isotropic coordinate system.
    kerr_lapse =
        std::sqrt(kerr_max(delta * sigma / kerr_A, 0.0));

    // Only the azimuthal shift component is nonzero in the local Kerr frame.
    spherical_shift[2] =
        -2.0 * mass * spin * r_bl / kerr_A;
}

#endif /* KERRBHINITIALDATA_IMPL_HPP_ */