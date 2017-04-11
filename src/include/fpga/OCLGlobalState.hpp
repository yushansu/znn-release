#ifndef __OCL_GLOBAL_STATE_H
#define __OCL_GLOBAL_STATE_H

#include <fstream>
#include "CL/opencl.h"

class cl_vars_t
{
  public:
    cl_int status;
    cl_uint num_platforms;
    cl_platform_id platform;
    cl_uint num_devices;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;

void cl_init(int platform_num, int device_num)
{
	 //const char * binary_fname = "/home/anderso2/fpga/GraphMat/bin/spmspv.aocx";

      // Get platform
      {
        status = clGetPlatformIDs(0, NULL, &num_platforms);
        assert(status == CL_SUCCESS);
        std::cout << "Found " << num_platforms << " platforms" << std::endl;
        if(platform_num >= num_platforms)
        {
          std::cout << "platform_num " << platform_num << " not available" << std::endl;
          exit(-1);
        }
        cl_platform_id * platform_ids = new cl_platform_id[num_platforms];
        status = clGetPlatformIDs(num_platforms,
                                  platform_ids,
    			      NULL);
        assert(status == CL_SUCCESS);
        for(cl_uint i = 0 ; i < num_platforms ; i++)
        {
          char str[255];
          status = clGetPlatformInfo(platform_ids[i],
                                     CL_PLATFORM_NAME,
      			         255,
    			         str,
    			         NULL);
          assert(status == CL_SUCCESS);
          std::cout << "Platform " << i << ": " << str << std::endl;
        }
        platform = platform_ids[platform_num];
        delete [] platform_ids;
      }
    
      // Get device
      {
        status = clGetDeviceIDs(platform,
                                CL_DEVICE_TYPE_ALL,
    			    0,
    			    NULL,
    			    &num_devices);
        assert(status == CL_SUCCESS);
        std::cout << "Found " << num_devices << " devices" << std::endl;
        if(device_num >= num_devices)
        {
          std::cout << "device_num " << device_num << " not available" << std::endl;
          exit(-1);
        }
        cl_device_id * device_ids = new cl_device_id[num_devices];
        status = clGetDeviceIDs(platform,
                                CL_DEVICE_TYPE_ALL,
    			    num_devices,
    			    device_ids,
    			    NULL);
        assert(status == CL_SUCCESS);
        for(cl_uint i = 0 ; i < num_devices ; i++)
        {
          char str[255];
          status = clGetDeviceInfo(device_ids[i],
                                   CL_DEVICE_NAME,
    			       255,
    			       str,
    			       NULL);
          std::cout << "Device " << i << ": " << str << std::endl;
        }
        device = device_ids[device_num];
        delete [] device_ids;
      }

      // Create context
      {
        cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, 
                                                      (cl_context_properties)platform,
      					          (cl_context_properties)0};
        context = clCreateContext(context_properties,
                                  1,
      			      &device,
    			      NULL,
    			      NULL,
    			      &status);
        assert(status == CL_SUCCESS);
      }
      /* Not yet needed */
      /*
      // Load binary
      {
        std::ifstream binfile;
        binfile.open(binary_fname, std::ios::binary| std::ios::in);
        if(!binfile.is_open())
        {
          std::cout << "File " << binary_fname << " did not open" << std::endl;
          exit(-1);
        }
        binfile.seekg(0, std::ios::end);
        size_t nbytes = binfile.tellg();
        binfile.seekg(0, std::ios::beg);
        char * binary_blob = new char[nbytes];
        binfile.read(binary_blob, nbytes);
        binfile.close();
    
        program = clCreateProgramWithBinary(context,
                                            1,
      				        &device,
    				        &nbytes,
    				        (const unsigned char**) &binary_blob,
    				        NULL,
    				        &status);
        assert(status == CL_SUCCESS);
        delete [] binary_blob;
    
        clBuildProgram(program,
                       1,
    		   &device,
    		   "-O2",
    		   NULL,
    		   NULL);
        assert(status == CL_SUCCESS);
      }
    
      // Create queue
      {
        queue = clCreateCommandQueue(context,
                                     device,
    				 CL_QUEUE_PROFILING_ENABLE,
    				 &status);
        assert(status == CL_SUCCESS);
      }
    
      // Create kernel
      {
        kernel = clCreateKernel(program,
                                "coospmspv",
    			    &status);
        assert(status == CL_SUCCESS);
      }
      */

}

void cl_finalize()
{
      //status = clReleaseKernel(kernel);
      //assert(status == CL_SUCCESS);
      //status = clReleaseProgram(program);
      //assert(status == CL_SUCCESS);
      status = clReleaseCommandQueue(queue);
      assert(status == CL_SUCCESS);
      status = clReleaseContext(context);
      assert(status == CL_SUCCESS);
}

};

extern cl_vars_t clv;

#endif
