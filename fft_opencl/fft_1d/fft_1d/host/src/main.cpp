#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
//#include "CL/cl_ext_altera.h"
#include "AOCLUtils/aocl_utils.h"

using namespace aocl_utils;

// Runtime constants
// Used to define the work set over which this kernel will execute.
static const size_t work_group_size = 8;  // 8 threads in the demo workgroup
// Defines kernel argument value, which is the workitem ID that will
// execute a printf call
static const int thread_id_to_output = 2;

// OpenCL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue1 = NULL;
static cl_command_queue queue2 = NULL;
static cl_command_queue queue3 = NULL;
static cl_kernel kernel1 = NULL;
static cl_kernel kernel2 = NULL;
static cl_kernel kernel3 = NULL;
static cl_program program = NULL;

// Function prototypes
bool init();
void cleanup();

// Entry point.
int main() {
  cl_int status;
  size_t sz = 1024;

  if(!init()) {
    return -1;
  }

  //////*****^_^^_^*******////////
  float * A = (float *)clSVMAllocAltera(context, CL_MEM_READ_ONLY, sz* sizeof(float),0);
  float * B = (float *)clSVMAllocAltera(context, CL_MEM_READ_ONLY, sz* sizeof(float),0);
  float * C = (float *)clSVMAllocAltera(context, CL_MEM_READ_ONLY, sz* sizeof(float),0);


    //Initialize SVM buffer
    for(size_t i = 0; i < sz; i++){
        A[i] = (float)i;
        B[i] = (float)(sz-i);
    }

    status = clSetKernelArgSVMPointer(kernel1, 0, (void*)A);
    checkError(status, "Failed to set kernel1 SVM pointer A");
    status = clSetKernelArgSVMPointer(kernel1, 1, (void*)B);
    checkError(status, "Failed to set kernel1 SVM pointer B");
    status = clSetKernelArgSVMPointer(kernel3, 0, (void*)C);
    checkError(status, "Failed to set kernel3 SVM pointer C");

    status = clEnqueueNDRangeKernel(queue1, kernel1, 1, NULL, &sz, NULL, 0, NULL, NULL);
    checkError(status, "Failed to Enqueue Kernel1");
    status = clEnqueueNDRangeKernel(queue2, kernel2, 1, NULL, &sz, NULL, 0, NULL, NULL);
    checkError(status, "Failed to Enqueue Kernel2");
    status = clEnqueueNDRangeKernel(queue3, kernel3, 1, NULL, &sz, NULL, 0, NULL, NULL);
    checkError(status, "Failed to Enqueue Kernel3");

    status = clFinish(queue1);
    checkError(status, "Failed to finish kernel1");
    status = clFinish(queue2);
    checkError(status, "Failed to finish kernel2");
    status = clFinish(queue3);
    checkError(status, "Failed to finish kernel3");
    printf("%d",(int)status);

    for(size_t i = 0; i < sz; i++){
        checkError(C[i] != (float)sz, "Error Result!");
    }

    clSVMFreeAltera(context, A);
    clSVMFreeAltera(context, B);
    clSVMFreeAltera(context, C);

    /***********


  // Set the kernel argument (argument 0)
  status = clSetKernelArg(kernel, 0, sizeof(cl_int), (void*)&thread_id_to_output);
  checkError(status, "Failed to set kernel arg 0");

  printf("\nKernel initialization is complete.\n");
  printf("Launching the kernel...\n\n");

  // Configure work set over which the kernel will execute
  size_t wgSize[3] = {work_group_size, 1, 1};
  size_t gSize[3] = {work_group_size, 1, 1};

  // Launch the kernel
  status = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, gSize, wgSize, 0, NULL, NULL);
  checkError(status, "Failed to launch kernel");

  // Wait for command queue to complete pending events
  status = clFinish(queue);
  checkError(status, "Failed to finish");
     ***********/

  printf("\nKernel execution is complete.\n");


  // Free the resources allocated
  cleanup();

  return 0;
}

/////// HELPER FUNCTIONS ///////

bool init() {
  cl_int status;

  if(!setCwdToExeDir()) {
    return false;
  }

  // Get the OpenCL platform.
  platform = findPlatform("Intel(R) FPGA");
  if(platform == NULL) {
    printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform.\n");
    return false;
  }

  // Query the available OpenCL devices.
  scoped_array<cl_device_id> devices;
  cl_uint num_devices;

  devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

  // We'll just use the first device.
  device = devices[0];

  //Get SVM capability

  //Check whether SVM is supported or not
  cl_device_svm_capabilities svm_capabilities;
  status = clGetDeviceInfo(device, CL_DEVICE_SVM_CAPABILITIES, sizeof(cl_device_svm_capabilities),&svm_capabilities,0);
  checkError(status,"No SVM support");

  //if(status == CL_SUCCESS && (svm_capabilities & CL_DEVICE_SVM_FINE_GRAIN_BUFFER)){
  //  printf("Fine-grained buffer");
  //}




  // Create the context.
  context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
  checkError(status, "Failed to create context");

  // Create the command queue.
  queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue2 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");
  queue3 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  checkError(status, "Failed to create command queue");

  // Create the program.
  std::string binary_file = getBoardBinaryFile("fft_1d", device);
  printf("Using AOCX: %s\n", binary_file.c_str());
  program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

  // Build the program that was just created.
  status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
  checkError(status, "Failed to build program");

  // Create the kernel - name passed in here must match kernel name in the
  // original CL file, that was compiled into an AOCX file using the AOC tool
  //const char *kernel_name = "fft_1d";  // Kernel name, as defined in the CL file
  kernel1 = clCreateKernel(program, "data_in", &status);
  checkError(status, "Failed to create kernel1");
  kernel2 = clCreateKernel(program, "fft_1d", &status);
  checkError(status, "Failed to create kernel2");
  kernel3 = clCreateKernel(program, "data_out", &status);
  checkError(status, "Failed to create kernel3");

  return true;
}

// Free the resources allocated during initialization
void cleanup() {
  if(kernel1) {
    clReleaseKernel(kernel1);
  }
  if(kernel2) {
    clReleaseKernel(kernel2);
  }
  if(kernel3) {
    clReleaseKernel(kernel3);
  }
  if(program) {
    clReleaseProgram(program);
  }
  if(queue1) {
    clReleaseCommandQueue(queue1);
  }
  if(queue2) {
    clReleaseCommandQueue(queue2);
  }
  if(queue3) {
    clReleaseCommandQueue(queue3);
  }
  if(context) {
    clReleaseContext(context);
  }
}
