/* GRTeclyn
 * Copyright 2022 The GRTL collaboration.
 * Please refer to LICENSE in GRTeclyn's root directory.
 */

#ifndef DIAGNOSTICSPARAMETERS_HPP_
#define DIAGNOSTICSPARAMETERS_HPP_

#include "GRParmParse.hpp"

//! How often (in level-0 steps) the diagnostics are written.
struct diagnostics_params_t
{
    int extraction_interval{1}; //!< x profiles (line extractions)
    int print_interval{1};      //!< constraint norms printed to stdout

    static void check_params()
    {
        GRParmParse pp("diagnostics");
        int extraction_interval{1};
        int print_interval{1};
        pp.queryAdd("extraction_interval", extraction_interval);
        pp.queryAdd("print_interval", print_interval);
        if (extraction_interval < 1)
        {
            pp.error("extraction_interval", "must be >= 1");
        }
        if (print_interval < 1)
        {
            pp.error("print_interval", "must be >= 1");
        }
    }

    void fill_params()
    {
        GRParmParse pp("diagnostics");
        pp.get("extraction_interval", extraction_interval);
        pp.get("print_interval", print_interval);
    }
};

#endif /* DIAGNOSTICSPARAMETERS_HPP_ */
