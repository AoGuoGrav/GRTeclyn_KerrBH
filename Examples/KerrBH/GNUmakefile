GRTECLYN_HOME = $(realpath ../..)

include $(GRTECLYN_HOME)/Tools/GNUMake/Make.defaults

EBASE = KerrBH

# Puncture tracking uses AMReX particles.
USE_PARTICLES = TRUE

AMREX_HOME ?= $(realpath ../../../amrex)

include $(GRTECLYN_HOME)/Tools/GNUMake/Make.defs

include $(AMREX_HOME)/Src/Base/Make.package
include $(AMREX_HOME)/Src/Boundary/Make.package
include $(AMREX_HOME)/Src/AmrCore/Make.package
include $(AMREX_HOME)/Src/Amr/Make.package
include $(AMREX_HOME)/Src/Particle/Make.package

GRTECLYN_SOURCE = $(GRTECLYN_HOME)/Source

# Directories containing source and header files required by the KerrBH example.
# The final dot means example-specific source files in this directory.
src_dirs := $(GRTECLYN_SOURCE)/BlackHoles \
            $(GRTECLYN_SOURCE)/CCZ4 \
            $(GRTECLYN_SOURCE)/Grids \
            $(GRTECLYN_SOURCE)/GRTeclynCore \
            $(GRTECLYN_SOURCE)/ParticleInterpolator \
            $(GRTECLYN_SOURCE)/IO \
            $(GRTECLYN_SOURCE)/Maths \
            $(GRTECLYN_SOURCE)/Tagging \
            $(realpath .)

# Collect source and header files from the required directories.
CEXE_sources += $(notdir $(foreach dir, $(src_dirs), \
                $(wildcard $(dir)/*.cpp)))

CEXE_headers += $(notdir $(foreach dir, $(src_dirs), \
                $(wildcard $(dir)/*.hpp)))

CEXE_headers += $(notdir $(foreach dir, $(src_dirs), \
                $(wildcard $(dir)/*.impl.hpp)))

VPATH_LOCATIONS += $(src_dirs)
INCLUDE_LOCATIONS += $(src_dirs)

include $(GRTECLYN_HOME)/Tools/GNUMake/Make.rules