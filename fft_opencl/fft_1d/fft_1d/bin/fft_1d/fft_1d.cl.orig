//ACL kernel for doing 1D FFT

channel float DATA_IN_X __attribute__((depth(8)));
channel float DATA_IN_Y __attribute__((depth(8)));
channel float DATA_OUT  __attribute__((depth(8)));

__kernel void data_in(__global const float *x, __global const float *y){
  //get index of work item
  int index = get_global_id(0);

  write_channel_altera(DATA_IN_X, x[index]);
  write_channel_altera(DATA_IN_Y, y[index]);
}

__kernel void fft_1d() {
  // it's just vector addition now
  float s = read_channel_altera(DATA_IN_X) + read_channel_altera(DATA_IN_Y);
  write_channel_altera(DATA_OUT,s);
}

__kernel void data_out(__global float *restrict z){
  int index = get_global_id(0);
  z[index] = read_channel_altera(DATA_OUT);
}

