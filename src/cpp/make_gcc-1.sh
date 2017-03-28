#!/bin/bash

if [[ $2 == "anderso2"* ]] ; then
AOCL_SDK=/home/anderso2/Projects/fpga/code/common
BOARD_DIR=/data1/bdw_fpga_pilot_opencl_bsp_v1.0/host/linux64/lib
BOARD_LIB=-lalterahalmmd
HLD_DIR=/data1/altera_pro/16.0/hld
BOOST_DIR=/usr/include/boost
FFTW_DIR=./
else
AOCL_SDK=/memex/yushans/common
BOARD_DIR=/nfs/yushans/intelFPGA/16.1/hld/board/s5_ref/linux64/lib
BOARD_LIB=-laltera_s5_ref_mmd
HLD_DIR=/nfs/yushans/intelFPGA/16.1/hld
BOOST_DIR=/nfs/yushans/boost_1_63_0
FFTW_DIR=/memex/yushans/fftw-3.3.6-pl1
fi

g++ -std=c++11 $1.cpp -I${BOOST_DIR} -I../../ -I${FFTW_DIR} -I../../src/include -DNDEBUG -O3 -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -lpthread -lrt -lfftw3f -o $1 -DZNN_DONT_CACHE_FFTS -O2 -fPIC -I../common/inc \
            -I${HLD_DIR}/host/include  ${AOCL_SDK}/src/AOCLUtils/opencl.cpp ${AOCL_SDK}/src/AOCLUtils/options.cpp -I${AOCL_SDK}/inc -L${BOARD_DIR}/linux64/lib -L${HLD_DIR}/host/linux64/lib -Wl,--no-as-needed -lalteracl ${BOARD_LIB} -lelf 

