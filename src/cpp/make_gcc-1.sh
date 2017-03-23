#!/bin/bash
g++ -std=c++11 $1.cpp -I/nfs/yushans/boost_1_63_0 -I../.. -I/memex/yushans/fftw-3.3.6-pl1 -I../../src/include -DNDEBUG -O3 -DZNN_CUBE_POOL_LOCKFREE -DZNN_USE_FLOATS -lpthread -lrt -lfftw3f -o $1 -DZNN_DONT_CACHE_FFTS
#-ljemalloc 
