/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef SIMULATIONPARAMETERS_HPP_
#define SIMULATIONPARAMETERS_HPP_

#include "BaseParameterChecker.hpp"
#include "CCZ4RHS.hpp"
#include "DiagnosticsParameters.hpp"
#include "EkpyroticBHData.hpp"
#include "FixedGridsTagger.hpp"
#include "IntegratedMovingPunctureGauge.hpp"
#include "LineExtractionParameters.hpp"
#include "Potential.hpp"
#include "ScalarField.hpp"
#include "XCTSParameters.hpp"

class SimulationParameters
{
  public:
    SimulationParameters() = delete;

    static void check_params()
    {
        BaseParameterChecker::check_params();
        CCZ4_params_t::check_params();
        IntegratedMovingPunctureGauge<
            FourthOrderDerivatives>::params_t::check_params();
        FixedGridsTagger::check_params();

        // The initial data need G_Newton, V0 and c, and the cosmological
        // data need the black hole parameters, so check these first
        ScalarField<Potential>::params_t::check_params();
        Potential::params_t::check_params();
        EkpyroticBHData::bh_params_t::check_params();
        EkpyroticBHData::cosmo_params_t::check_params();
        xcts_params_t::check_params();
        diagnostics_params_t::check_params();

        line_extraction_params_t::check_params("chi_line_extraction");
        line_extraction_params_t::check_params("lapse_line_extraction");
        line_extraction_params_t::check_params("xcts_line_extraction");
        line_extraction_params_t::check_params("constraints_line_extraction");
    }
};

#endif /* SIMULATIONPARAMETERS_HPP_ */
