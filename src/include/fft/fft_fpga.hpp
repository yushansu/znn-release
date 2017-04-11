//
// Copyright (C) 2012-2015  Aleksandar Zlateski <zlateski@mit.edu>
// ---------------------------------------------------------------
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.
//
#pragma once

#include "fftw_plans.hpp"
#include "fft_fpga_transform.hpp"

#include <zi/time.hpp>

#ifdef ZNN_MEASURE_FFT_RUNTIME
#  define ZNN_MEASURE_FFT_START() zi::wall_timer wt
#  define ZNN_MEASURE_FFT_END() fftw_stats.add(wt.elapsed<double>())
#else
#  define ZNN_MEASURE_FFT_START() static_cast<void>(0)
#  define ZNN_MEASURE_FFT_END() static_cast<void>(0)
#endif

namespace znn { namespace v4 {

class fft_stats_impl
{
private:
    double              total_time_;
    std::size_t         total_     ;
    mutable std::mutex  m_         ;

public:
    fft_stats_impl()
        : total_time_(0)
        , total_(0)
        , m_()
    { }

    double get_total_time() const
    {
        guard g(m_);
        return total_time_;
    }

    void reset_total_time()
    {
        guard g(m_);
        total_time_ = 0;
    }

    size_t get_total() const
    {
        guard g(m_);
        return total_;
    }

    void add(double time)
    {
        guard g(m_);
        ++total_;
        total_time_ += time;
    }
};

namespace {
fft_stats_impl& fft_stats = zi::singleton<fft_stats_impl>::instance();
} // anonymous namespace

class fftw
{
public:
    class transformer
    {
    private:
        vec3i    sz           ;
        vec3i    actual_sz    ;

    public:
        transformer(const vec3i& s)
            : sz(s)
            , actual_sz(s)
        {
        }

        vec3i const & size() const
        {
            return sz;
        }

        vec3i const & actual_size() const
        {
            return actual_sz;
        }

        void forward( cube<real>& in,
                      cube<complex>& out )
        {
            ZI_ASSERT(v4::size(out)==fft_complex_size(in));
            ZI_ASSERT(v4::size(in)==actual_sz);

            ZNN_MEASURE_FFT_START();
            forward_r2c_3d_fpga(sz.get<0>(),
                                sz.get<1>(),
                                sz.get<2>(),
                                reinterpret_cast<real*>(in.data()),
                                reinterpret_cast<complex*>(out.data())
                                );
            ZNN_MEASURE_FFT_END();
        }

        void backward( cube<complex>& in,
                       cube<real>& out )
        {
            ZI_ASSERT(v4::size(in)==fft_complex_size(out));
            ZI_ASSERT(v4::size(out)==actual_sz);

            ZNN_MEASURE_FFT_START();
            backward_c2r_3d_fpga(sz.get<0>(),
                                 sz.get<1>(),
                                 sz.get<2>(),
                                 reinterpret_cast<complex*>(in.data()),
                                 reinterpret_cast<real*>(out.data()));
            ZNN_MEASURE_FFT_END();
        }

        cube_p<complex> forward( cube_p<real>&& in )
        {
            cube_p<complex> ret = get_cube<complex>(fft_complex_size(*in));
            forward( *in, *ret );
            return ret;
        }

        cube_p<complex> forward_pad( const ccube_p<real>& in )
        {
            cube_p<real> pin = pad_zeros(*in, actual_sz);
            return forward(std::move(pin));
        }

        cube_p<real> backward( cube_p<complex>&& in )
        {
            cube_p<real> ret = get_cube<real>(actual_sz);
            backward( *in, *ret );
            return ret;
        }
    };


public:
    static void forward( cube<real>& in,
                         cube<complex>& out )
    {
        ZI_ASSERT(in.shape()[0]==out.shape()[0]);
        ZI_ASSERT(in.shape()[1]==out.shape()[1]);
        ZI_ASSERT((in.shape()[2]/2+1)==out.shape()[2]);

        ZNN_MEASURE_FFT_START();
        forward_r2c_3d_fpga(in.shape()[0],
                            in.shape()[1],
                            in.shape()[2],
                            reinterpret_cast<real*>(in.data()),
                            reinterpret_cast<complex*>(out.data()));
        ZNN_MEASURE_FFT_END();
    }

    static void backward( cube<complex>& in,
                          cube<real>& out )
    {
        ZI_ASSERT(in.shape()[0]==out.shape()[0]);
        ZI_ASSERT(in.shape()[1]==out.shape()[1]);
        ZI_ASSERT((out.shape()[2]/2+1)==in.shape()[2]);

        ZNN_MEASURE_FFT_START();
        backward_c2r_3d_fpga(in.shape()[0],
                             in.shape()[1],
                             in.shape()[2],
                             reinterpret_cast<complex*>(in.data()),
                             reinterpret_cast<real*>(out.data()));
        ZNN_MEASURE_FFT_END();
    }

    static cube_p<complex> forward( cube_p<real>&& in )
    {
        cube_p<complex> ret = get_cube<complex>(fft_complex_size(*in));
        fftw::forward( *in, *ret );
        return ret;
    }

    static cube_p<real> backward( cube_p<complex>&& in, const vec3i& s )
    {
        cube_p<real> ret = get_cube<real>(s);
        fftw::backward( *in, *ret );
        return ret;
    }

    static cube_p<complex> forward_pad( const ccube_p<real>& in,
                                        const vec3i& pad )
    {
        cube_p<real> pin = pad_zeros(*in, pad);
        return fftw::forward(std::move(pin));
    }

}; // class fftw

}} // namespace znn::v4

#undef ZNN_MEASURE_FFT_START
#undef ZNN_MEASURE_FFT_END

#undef FFT_EXECUTE_DFT_R2C
#undef FFT_EXECUTE_DFT_C2R

