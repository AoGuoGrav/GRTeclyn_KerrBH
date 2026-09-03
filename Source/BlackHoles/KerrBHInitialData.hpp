/* GRTeclyn
 *
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef KERRBHINITIALDATA_HPP_
#define KERRBHINITIALDATA_HPP_

#include "CCZ4Vars.hpp"
#include "Coordinates.hpp"
#include "DimensionDefinitions.hpp"

#include <AMReX_Array4.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_IntVect.H>
#include <AMReX_REAL.H>

#include <array>

static_assert(AMREX_SPACEDIM == 3,
              "KerrBHInitialData is implemented only in three dimensions.");

class KerrBHInitialData
{
public:
    struct params_t
    {
        amrex::Real mass{};
        amrex::Real spin{};

        std::array<amrex::Real, AMREX_SPACEDIM> center{};
        std::array<amrex::Real, AMREX_SPACEDIM> spin_direction{
            0.0, 0.0, 1.0};

        void check_params();
        void fill_params();
    };

    explicit KerrBHInitialData(amrex::Real a_dx);

    AMREX_GPU_DEVICE
    void operator()(int ix, int iy, int iz,
                    const amrex::Array4<amrex::Real> &state) const;

private:
    using vector_t = std::array<amrex::Real, AMREX_SPACEDIM>;
    using matrix_t = std::array<vector_t, AMREX_SPACEDIM>;

    amrex::Real m_dx;
    params_t m_params;

    AMREX_GPU_DEVICE
    void compute_kerr(matrix_t &spherical_g, matrix_t &spherical_K,
                      vector_t &spherical_shift, amrex::Real &kerr_lapse,
                      const vector_t &coords) const;
};

#include "KerrBHInitialData.impl.hpp"

#endif /* KERRBHINITIALDATA_HPP_ */