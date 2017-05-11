#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <cstring>
#include "CL/opencl.h"
//#include "CL/cl_ext_altera.h"
#include "AOCLUtils/aocl_utils.h"
#include "../inc/fft_config.h"
//#include "/memex/yushans/light-matrix/light_mat/matrix/matrix_transpose.h"
//#include "/Users/suyushan/Documents/FPGA/light-matrix/light_mat/matrix"
//#include "bench/bench_base.h"

// the above header defines log of the FFT size hardcoded in the kernel
// compute N as 2^LOGN
#define N (1 << LOGN)

//#define USE_SVM_API 1

using namespace aocl_utils;

#define STRING_BUFFER_LEN 1024

// ACL runtime configuration
static cl_platform_id platform = NULL;
static cl_device_id device = NULL;
static cl_context context = NULL;
static cl_command_queue queue = NULL;
static cl_command_queue queue1 = NULL;
static cl_kernel kernel = NULL;
static cl_kernel kernel1 = NULL;
static cl_program program = NULL;
static cl_int status = 0;

// FFT operates with complex numbers - store them in a struct
typedef struct {
    double x;
    double y;
} double2;

typedef struct {
    float x;
    float y;
} float2;

// Function prototypes
bool init();
void cleanup();
static void test_fft(int iterations, bool inverse);
static int coord(int iteration, int i);
static void fourier_transform_gold(bool inverse, int lognr_points, double2 * data);
static void fourier_transform_gold_3D(bool inverse, int lognr_points, double2 * data, int n);
static void fourier_stage(int lognr_points, double2 * data);
static void fft_1D_FPGA(float2 * h_inData, int iterations, bool inverse);

//#define IDX(x,y,z,n) ( ( x ) + ( y )*( n ) + ( z ) * ( n ) * ( n ))
#define IDX(x,y,z,n) ( ( z ) + ( y )*( n ) + ( x ) * ( n ) * ( n ))


// Host memory buffers
float2 *h_inData, *h_outData, *output;
double2 *h_verify;

// Entry point.
int main(int argc, char **argv) {
    if(!init()) {
        return false;
    }
    int iterations = N * N;

    Options options(argc, argv);

    // Optional argument to set the number of iterations.
    if(options.has("n")) {
        iterations = options.get<int>("n");
    }

    if (iterations <= 0) {
        printf("ERROR: Invalid number of iterations\n\nUsage: %s [-N=<#>]\n\tN: number of iterations to run (default 2000)\n", argv[0]);
        return false;
    }

    h_inData = (float2 *)clSVMAllocAltera(context, CL_MEM_READ_WRITE, sizeof(float2) * N * N * N, 0);
    h_outData = (float2 *)clSVMAllocAltera(context, CL_MEM_READ_WRITE, sizeof(float2) * N * N * N, 0);

    h_verify = (double2 *)alignedMalloc(sizeof(double2) * N * N * N);
    if (!(h_inData && h_outData && h_verify)) {
        printf("ERROR: Couldn't create host buffers\n");
        return false;
    }

    test_fft(iterations, false); // test FFT transform running a sequence of iterations x 4k points transforms
    //test_fft(iterations, true); // test inverse FFT transform - same setup as above

    // Free the resources allocated
    cleanup();

    return 0;
}

void test_fft(int iterations, bool inverse) {
    //printf("Launching");
    if (inverse)
        printf(" inverse");
    //printf(" FFT transform for %d iterations\n", iterations);


    // Initialize input and produce verification data
    for (int i = 0; i < 1; i++) {
        for (int j = 0; j < N * N * N; j++) {
            h_verify[coord(i, j)].x = h_inData[coord(i, j)].x = (float)((double)rand() / (double)RAND_MAX);
            h_verify[coord(i, j)].y = h_inData[coord(i, j)].y = (float)((double)rand() / (double)RAND_MAX);
        }
    }
/*
    for(int j = 0; j < N * N * N; j++){
        printf("%f\n", h_inData[j].x);
    }

    for(int j = 0; j < N * N * N; j++){
        printf("%f\n", h_inData[j].y);
    }
*/
    // Can't pass bool to device, so convert it to int
    int inverse_int = inverse;

    // Get the iterationstamp to evaluate performance
    double time = getCurrentTimestamp();
    // Set the kernel arguments

 //   float2 *temp_vec = (float2 *)clSVMAllocAltera(context, CL_MEM_READ_ONLY, sizeof(float2) * N * iterations, 0);
 //   checkError(status, "Fail to alloc temp_vec");
    /*
    for(int x = 0; x < N; x ++) {
        for (int y = 0; y < N; y++) {
            for (int z = 0; z < N; z++) {
                temp_vec[x * N * N + y * N + z].x = h_inData[IDX(x, y, z, N)].x;
                temp_vec[x * N * N + y * N + z].y = h_inData[IDX(x, y, z, N)].y;
            }
        }
    }
    */
    //time = getCurrentTimestamp() - time;

    //printf("copy\n");

    //printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));

    //time = getCurrentTimestamp();

    fft_1D_FPGA(h_inData, iterations, inverse);

    //time = getCurrentTimestamp() - time;

    //printf("1st fft\n");

    //printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));

    //time = getCurrentTimestamp();
/*
    for(int x = 0; x < N; x++){
        for(int y = 0; y < N; y++){
            for(int z = 0; z < N; z++){
                h_inData[IDX(x, y, z, N)].x = h_outData[x * N * N + y * N + z].x;
                h_inData[IDX(x, y, z, N)].y = h_outData[x * N * N + y * N + z].y;
            }
        }
    }


    for(int y = 0; y < N; y ++) {
        for (int z = 0; z < N; z++) {
            for (int x = 0; x < N; x++) {
                temp_vec[y * N * N + z * N + x].x = h_inData[IDX(x, y, z, N)].x;
                temp_vec[y * N * N + z * N + x].y = h_inData[IDX(x, y, z, N)].y;
            }
        }
    }

*/


    for(int x = 0; x < N; x++){
        for(int y = 0; y < N; y++){
            for(int z = 0; z < N; z++){
                h_inData[y * N * N + z * N + x].x = h_outData[x * N * N + y * N + z].x;
                h_inData[y * N * N + z * N + x].y = h_outData[x * N * N + y * N + z].y;
            }
        }
    }


    //time = getCurrentTimestamp() - time;

    //printf("transpose\n");

    //printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));

    //time = getCurrentTimestamp();

    fft_1D_FPGA(h_inData, iterations, inverse);

    //time = getCurrentTimestamp() - time;

    //printf("2nd fft\n");

    //printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));

    //time = getCurrentTimestamp();
/*
    for(int y = 0; y < N; y++){
        for(int z = 0; z < N; z++){
            for(int x = 0; x < N; x++){
                h_inData[IDX(x, y, z, N)].x = h_outData[y * N * N + z * N + x].x;
                h_inData[IDX(x, y, z, N)].y = h_outData[y * N * N + z * N + x].y;
            }
        }
    }

    for(int x = 0; x < N; x ++) {
        for (int z = 0; z < N; z++) {
            for (int y = 0; y < N; y++) {
                temp_vec[x * N * N + z * N + y].x = h_inData[IDX(x, y, z, N)].x;
                temp_vec[x * N * N + z * N + y].y = h_inData[IDX(x, y, z, N)].y;
            }
        }
    }
*/

    for(int y = 0; y < N; y++){
        for(int z = 0; z < N; z++){
            for(int x = 0; x < N; x++){
                h_inData[x * N * N + z * N + y].x = h_outData[y * N * N + z * N + x].x;
                h_inData[x * N * N + z * N + y].y = h_outData[y * N * N + z * N + x].y;
            }
        }
    }


    //time = getCurrentTimestamp() - time;

    //printf("transpose\n");

    //printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));

    //time = getCurrentTimestamp();

    fft_1D_FPGA(h_inData, iterations, inverse);


    //time = getCurrentTimestamp() - time;

    //printf("third fft\n");

    //printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));

    //time = getCurrentTimestamp();

    for(int x = 0; x < N; x++){
        for(int z = 0; z < N; z++){
            for(int y = 0; y < N; y++){
                h_inData[IDX(x, y, z, N)].x = h_outData[x * N * N + z * N + y].x;
                h_inData[IDX(x, y, z, N)].y = h_outData[x * N * N + z * N + y].y;
            }
        }
    }


    // Record execution time
    time = getCurrentTimestamp() - time;

    printf("transpose\n");

    printf("\tProcessing time = %.4fms\n", (float)(time * 1E3));
    double gpoints_per_sec = ((double) iterations * N / time) * 1E-9;
    double gflops = 5 * N * (log((float)N)/log((float)2))/(time / iterations * 1E9);
    printf("\tThroughput = %.4f Gpoints / sec (%.4f Gflops)\n", gpoints_per_sec, gflops);
/*
    for (int i = 0; i < N * N * N; i++) {
        h_outData[i] = h_inData[i];
    }
    */
/*
    for(int j = 0; j < N * N * N; j++){
        printf("%f\n", h_inData[j].x);
    }

    for(int j = 0; j < N * N * N; j++){
        printf("%f\n", h_inData[j].y);
    }
*/
    // Print as imaginary number
    //for (int i = 0; i < N * N * N; i++) {
    //    printf("%f + %fi\n", h_outData[i].x, h_outData[i].y);
    //}
    // Pick randomly a few iterations and check SNR
/*
    double fpga_snr = 200;
    for (int i = 0; i < iterations; i+= rand() % 20 + 1) {
        fourier_transform_gold(inverse, LOGN, h_verify + coord(i, 0));
        double mag_sum = 0;
        double noise_sum = 0;
        for (int j = 0; j < N; j++) {
            double magnitude = (double)h_verify[coord(i, j)].x * (double)h_verify[coord(i, j)].x +
                               (double)h_verify[coord(i, j)].y * (double)h_verify[coord(i, j)].y;
            double noise = (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) * (h_verify[coord(i, j)].x - (double)h_outData[coord(i, j)].x) +
                           (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y) * (h_verify[coord(i, j)].y - (double)h_outData[coord(i, j)].y);

            mag_sum += magnitude;
            noise_sum += noise;
        }
        double db = 10 * log(mag_sum / noise_sum) / log(10.0);
        // find minimum SNR across all iterations
        if (db < fpga_snr) fpga_snr = db;
    }

    printf("\tSignal to noise ratio on output sample: %f --> %s\n\n", fpga_snr, fpga_snr > 125 ? "PASSED" : "FAILED");
    */
}

/////// HELPER FUNCTIONS ///////

// provides a linear offset in the input array
int coord(int iteration, int i) {
    return iteration * N + i;
}

bool init() {
    cl_int status;

    if(!setCwdToExeDir()) {
        return false;
    }

    // Get the OpenCL platform.
    platform = findPlatform("Altera");
    if(platform == NULL) {
        printf("ERROR: Unable to find Intel(R) FPGA OpenCL platform\n");
        return false;
    }

    // Query the available OpenCL devices.
    scoped_array<cl_device_id> devices;
    cl_uint num_devices;

    devices.reset(getDevices(platform, CL_DEVICE_TYPE_ALL, &num_devices));

    // We'll just use the first device.
    device = devices[0];

    // Create the context.
    context = clCreateContext(NULL, 1, &device, &oclContextCallback, NULL, &status);
    checkError(status, "Failed to create context");

    // Create the command queue.
    queue = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");
    queue1 = clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
    checkError(status, "Failed to create command queue");

    // Create the program.
    std::string binary_file = getBoardBinaryFile("fft1d", device);
    //printf("Using AOCX: %s\n\n", binary_file.c_str());
    program = createProgramFromBinary(context, binary_file.c_str(), &device, 1);

    // Build the program that was just created.
    status = clBuildProgram(program, 0, NULL, "", NULL, NULL);
    checkError(status, "Failed to build program");

    // Create the kernel - name passed in here must match kernel name in the
    // original CL file, that was compiled into an AOCX file using the AOC tool
    kernel = clCreateKernel(program, "fft1d", &status);
    checkError(status, "Failed to create kernel");

    kernel1 = clCreateKernel(program, "fetch", &status);
    checkError(status, "Failed to create fetch kernel");

    cl_device_svm_capabilities caps = 0;

    status = clGetDeviceInfo(
            device,
            CL_DEVICE_SVM_CAPABILITIES,
            sizeof(cl_device_svm_capabilities),
            &caps,
            0
    );
    checkError(status, "Failed to get device info");

    if (!(caps & CL_DEVICE_SVM_COARSE_GRAIN_BUFFER)) {
        printf("The host was compiled with USE_SVM_API, however the device currently being targeted does not support SVM.\n");
        // Free the resources allocated
        cleanup();
        return false;
    }

    return true;
}

// Free the resources allocated during initialization
void cleanup() {
    if(kernel)
        clReleaseKernel(kernel);
    if(program)
        clReleaseProgram(program);
    if(queue)
        clReleaseCommandQueue(queue);
    if (h_verify)
        alignedFree(h_verify);

    if (h_inData)
        clSVMFreeAltera(context, h_inData);
    if (h_outData)
        clSVMFreeAltera(context, h_outData);

    if(context)
        clReleaseContext(context);
}

void fourier_stage(int lognr_points, double2 *data) {
    int nr_points = 1 << lognr_points;
    if (nr_points == 1) return;
    double2 *half1 = (double2 *)alloca(sizeof(double2) * nr_points / 2);
    double2 *half2 = (double2 *)alloca(sizeof(double2) * nr_points / 2);
    for (int i = 0; i < nr_points / 2; i++) {
        half1[i] = data[2 * i];
        half2[i] = data[2 * i + 1];
    }
    fourier_stage(lognr_points - 1, half1);
    fourier_stage(lognr_points - 1, half2);
    for (int i = 0; i < nr_points / 2; i++) {
        data[i].x = half1[i].x + cos (2 * M_PI * i / nr_points) * half2[i].x + sin (2 * M_PI * i / nr_points) * half2[i].y;
        data[i].y = half1[i].y - sin (2 * M_PI * i / nr_points) * half2[i].x + cos (2 * M_PI * i / nr_points) * half2[i].y;
        data[i + nr_points / 2].x = half1[i].x - cos (2 * M_PI * i / nr_points) * half2[i].x - sin (2 * M_PI * i / nr_points) * half2[i].y;
        data[i + nr_points / 2].y = half1[i].y + sin (2 * M_PI * i / nr_points) * half2[i].x - cos (2 * M_PI * i / nr_points) * half2[i].y;
    }
}

// Reference Fourier transform
void fourier_transform_gold(bool inverse, const int lognr_points, double2 *data) {
    const int nr_points = 1 << lognr_points;

    // The inverse requires swapping the real and imaginary component

    if (inverse) {
        for (int i = 0; i < nr_points; i++) {
            double tmp = data[i].x;
            data[i].x = data[i].y;
            data[i].y = tmp;;
        }
    }
    // Do a FT recursively
    fourier_stage(lognr_points, data);

    // The inverse requires swapping the real and imaginary component
    if (inverse) {
        for (int i = 0; i < nr_points; i++) {
            double tmp = data[i].x;
            data[i].x = data[i].y;
            data[i].y = tmp;;
        }
    }
}

void fourier_transform_gold_3D(bool inverse, const int lognr_points, double2 *data, int n){
    double2 *temp_vec1 = (double2 *)alloca(sizeof(double2) * n);
    for(int i = 0; i < 10; i++){
        printf("%f %f\n", data[i].x, data[i].y);
    }
    for(int x = 0; x < n; x ++){
        for(int y = 0; y < n; y++){
            for(int z = 0; z < n; z++){
                temp_vec1[z].x = data[IDX(x, y, z, n)].x;
                temp_vec1[z].y = data[IDX(x, y, z, n)].y;
            }
            fourier_transform_gold(inverse, lognr_points, temp_vec1);
            for(int z = 0; z < n; z++){
                data[IDX(x, y, z, n)].x = (float)temp_vec1[z].x;
                data[IDX(x, y, z, n)].y = (float)temp_vec1[z].y;
            }
        }
    }

    printf("Finished 1st dimension");
    for(int y = 0; y < n; y ++){
        for(int z = 0; z < n; z++){
            for(int x = 0; x < n; x++){
                temp_vec1[x] = data[IDX(x, y, z, n)];
            }
            fourier_transform_gold(inverse, lognr_points, temp_vec1);
            for(int x = 0; x < n; x++){
                data[IDX(x, y, z, n)] = temp_vec1[x];
            }
        }
    }

    printf("Finished 2nd dimension");
    for(int x = 0; x < n; x ++){
        for(int z = 0; z < n; z++){
            for(int y = 0; y < n; y++){
                temp_vec1[y] = data[IDX(x, y, z, n)];
            }
            fourier_transform_gold(inverse, lognr_points, temp_vec1);
            for(int y = 0; y < n; y++){
                data[IDX(x, y, z, n)] = temp_vec1[y];
            }
        }
    }

    printf("Finished 3rd dimension");
}

void fft_1D_FPGA(float2 * input, int iterations, bool inverse){
    int inverse_int = inverse;
    //output = (float2 *)clSVMAllocAltera(context, CL_MEM_READ_WRITE, sizeof(float2) * N * iterations, 0);
    status = clSetKernelArgSVMPointer(kernel1, 0, (void *)input);
    checkError(status, "Failed to set kernel1 arg 0");

    status = clSetKernelArgSVMPointer(kernel, 0, (void *)h_outData);

    checkError(status, "Failed to set kernel arg 0");
    status = clSetKernelArg(kernel, 1, sizeof(cl_int), (void*)&iterations);
    checkError(status, "Failed to set kernel arg 1");
    status = clSetKernelArg(kernel, 2, sizeof(cl_int), (void*)&inverse_int);
    checkError(status, "Failed to set kernel arg 2");

    //printf(inverse ? "\tInverse FFT" : "\tFFT");
    //printf(" kernel initialization is complete.\n");

    // Launch the kernel - we launch a single work item hence enqueue a task
    status = clEnqueueTask(queue, kernel, 0, NULL, NULL);
    checkError(status, "Failed to launch kernel");

    size_t ls = N/8;
    size_t gs = iterations * ls;
    status = clEnqueueNDRangeKernel(queue1, kernel1, 1, NULL, &gs, &ls, 0, NULL, NULL);
    checkError(status, "Failed to launch fetch kernel");

    // Wait for command queue to complete pending events
    status = clFinish(queue);
    checkError(status, "Failed to finish");
    status = clFinish(queue1);
    checkError(status, "Failed to finish queue1");

    // bit reversing makes it match up with the python validate script
    int nr_points = N;
    int lognr_points = LOGN;
    float2 *temp = (float2 *)alloca(sizeof(float2) * nr_points * iterations);
    for(int n = 0; n < N; n++) {
        for (int m = 0; m < N; m++) {
            for (int i = 0; i < nr_points; i++) temp[i] = h_outData[n * nr_points * nr_points + m * nr_points + i];
            for (int i = 0; i < nr_points; i++) {
                int fwd = i;
                int bit_rev = 0;
                for (int j = 0; j < lognr_points; j++) {
                    bit_rev <<= 1;
                    bit_rev |= fwd & 1;
                    fwd >>= 1;
                }
                h_outData[n * nr_points * nr_points + m * nr_points + i] = temp[bit_rev];
            }
        }
    }
}
