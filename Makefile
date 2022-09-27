##############################################################################
##
## Copyright (c) 2018 Mellanox Technologies LTD. All rights reserved.
##
## This program is free software; you can redistribute it and#or
## modify it under the terms of the GNU Lesser General Public License
## as published by the Free Software Foundation; either version 2
## of the License, or (at your option) any later version.
##
## You should have received a copy of the GNU Lesser General Public License
## along with this program; if not, see <http:##www.gnu.org#licenses#>.
##
##     Redistribution and use in source and binary forms, with or
##     without modification, are permitted provided that the following
##     conditions are met:
##
##      - Redistributions of source code must retain the above
##        copyright notice, this list of conditions and the following
##        disclaimer.
##
##      - Redistributions in binary form must reproduce the above
##        copyright notice, this list of conditions and the following
##        disclaimer in the documentation and#or other materials
##        provided with the distribution.
##
## THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
## EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
## MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
## NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
## BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
## ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
## CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
## SOFTWARE.
##
##############################################################################

MODE ?= debug
EXT_RELEASE ?=0

ifdef CCQ_GEM5
   GEM5_SIMUL = 1
else
   GEM5_SIMUL ?= 0
endif

GCC_INSTALL ?= /auto/sw_tools/OpenSource/gcc/INSTALLS/gcc-4.9.3/linux_x86_64
PYTHON_INSTALL ?=  /auto/sw_tools/OpenSource/python/INSTALLS/python_2.7.11/linux_x86_64

CCQ_GEM5 ?= /auto/sw/projects/arch_modelling/soc_model/AravaCC/gem5_model/RELEASE/latest
CCQ_GEM5_DIR = $(abspath ${CCQ_GEM5})

MKFILE_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))

export PATH:=${GCC_INSTALL}/bin:${PATH}

ifeq (${MODE},release)
GEM5LIB=gem5_opt
DEXT=rel
CXXFLAGS  = "-Wall --std=c++11"
else
GEM5LIB=gem5_debug
DEXT=dbg
CXXFLAGS  = "-Wall --std=c++11 -ggdb  -DDEBUG=1"
DEFINES  += "-DDEBUG=1"
endif

ifeq (${EXT_RELEASE},1)
   GEM5LIB                 = arava_cc_irisc
   GEM5_NOPYTHON           = 1 
   GEM5_ENCRYPTED_CONFIG   = "./mlnx_cc_configdata.h"
   DEFINES                += "-DEXTERNAL_RELEASE"
   LIBDIRS   += /usr/cad/omnet/omnest-4.6-zed/lib/
endif

INCDIRS   += ../FW
INCDIRS   += ../FW/include
INCDIRS   += ../FW/Algorithm
INCDIRS   += ../FW/Mellanox
INCDIRS   += ../FW/include/platform

INCDIRS   += ../DCTrafficGen/src
LIBDIRS   += ./DCTrafficGen/out/gcc-$$\(MODE\)/src/
LIBS      += DCTG

INCDIRS   += ${abspath ${CCQ_GEM5_DIR}/MellanoxModels/CoSim}
INCDIRS   += ${abspath ${CCQ_GEM5_DIR}/MellanoxModels/OmnetCoSim}
INCDIRS   += ${abspath ${CCQ_GEM5_DIR}/MellanoxModels/AravaCC/ccSOC}

INCDIRS   += /usr/include/libxml2/
# LIBS      += example_$$\(MODE\)

LIBDIRS   += $(MKFILE_DIR)/out/$$\(CONFIGNAME\)/algo
LIBDIRS   += ${GCC_INSTALL}/lib64

ifeq (${GEM5_SIMUL}, 1)
   LIBDIRS   += ${abspath ${CCQ_GEM5_DIR}/gem5/build/ARAVA_CC_IRISC}
   LIBS      += ${GEM5LIB}
   ifdef GEM5_NOPYTHON
      DEFINES   += -DGEM5_NOPYTHON
      ifdef GEM5_ENCRYPTED_CONFIG
         DEFINES   += -DGEM5_ENCRYPTED_CONFIG=${GEM5_ENCRYPTED_CONFIG}
      endif
      LIBS      += z crypto
   else
      LIBDIRS   += ${abspath ${PYTHON_INSTALL}/lib}
   endif
   LIBS      += pthread
   DEFINES   += -DGEM5COSIM=1
   DEFINES   += -DCCQ_GEM5=${CCQ_GEM5_DIR}
endif

all: checkmakefiles
	cd src && $(MAKE) CXXFLAGS=$(CXXFLAGS) 

param_only: checkmakefiles
	cd src && $(MAKE) CXXFLAGS="$(CXXFLAGS) -DNO_GEM5=1"

clean: checkmakefiles
	cd src && $(MAKE) clean 

cleanall: checkmakefiles 
	cd src && $(MAKE) MODE=release clean 
	cd src && $(MAKE) MODE=debug clean 

makefiles:
	cd src &&                                                         \
           opp_makemake                                                   \
             -P ${PWD}                                                    \
             -M ${MODE}                                                   \
             -O out                                                       \
             -f --deep                                                     \
             -o ccsim                                                     \
             ${DEFINES}                                                   \
             $(foreach i, $(INCDIRS), $(addprefix -I, $i))                \
             $(foreach l, $(LIBDIRS), $(addprefix -L, $(abspath $l)))     \
             $(foreach l, $(LIBS), $(addprefix -l, $l))

makefilesmakeso:
	cd src &&                                                         \
           opp_makemake                                                   \
             -P ${PWD}                                                    \
             -M ${MODE}                                                   \
             -O out                                                       \
             -f --deep --make-so                                                     \
             -o ccsim                                                     \
             ${DEFINES}                                                   \
             $(foreach i, $(INCDIRS), $(addprefix -I, $i))                \
             $(foreach l, $(LIBDIRS), $(addprefix -L, $(abspath $l)))     \
             $(foreach l, $(LIBS), $(addprefix -l, $l))

checkmakefiles: 
	@if [ ! -f src/Makefile ]; then \
      echo; \
           echo '=======================================================================';  \
           echo 'src/Makefile does not exist. Please use "make makefiles" to generate it!'; \
           echo '=======================================================================';  \
           echo; \
           exit 1; \
	fi 

