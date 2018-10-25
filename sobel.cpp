#include <vector>
#include "sobel.h"
#include <stdio.h>
#include <errno.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


const char *kernel = 
"void __kernel find_edge(global unsigned char *in, global unsigned char *out,\n"
"const unsigned int w, const unsigned int h) {\n"
"size_t id = get_global_id(0);\n"
"size_t size = get_global_size(0);\n"
"size_t y = id / w;\n"
"size_t x = id % w;\n"
"if(x == 0 || x == w - 1 || y == 0 || y == h - 1) return;\n"
"// Compute gradient in +ve x direction\n"
"float gradient_X = in[ (x-1) + (y-1) * w ]\n"
"- in[ (x+1) + (y-1) * w ]\n"
"+ 2 * in[ (x-1) +  y    * w ]\n"
"- 2 * in[ (x+1) +  y    * w ]\n"
"+ in[ (x-1) + (y+1) * w ]\n"
"- in[ (x+1) + (y+1) * w ];\n"
"// Compute gradient in +ve y direction\n"
"float gradient_Y = in[ (x-1) + (y-1) * w ]\n"
"+ 2 * in[  x    + (y-1) * w ]\n"
"+ in[ (x+1) + (y-1) * w ]\n"
"- in[ (x-1) + (y+1) * w ]\n"
"- 2 * in[  x    + (y+1) * w ]\n"
"- in[ (x+1) + (y+1) * w ];\n"
"int value = ceil(sqrt(pow(gradient_X, 2) + pow(gradient_Y, 2)));\n"
"out[id] = 255 - value;\n"
"}\n";

void handle(int line, cl_int err) {
  if(err == CL_SUCCESS) return;
  printf("Error at line %d, with %d error code\n", line, err);
  exit(err);
}

cl_int get_platform(const char *vendor, cl_platform_id *chosen) {
  cl_uint num_platforms;
  cl_int ret;

  ret = clGetPlatformIDs(0, NULL, &num_platforms);
  if(ret != CL_SUCCESS) return ret;
  
  cl_platform_id *platforms = (cl_platform_id*)calloc(num_platforms, sizeof(cl_platform_id));

  ret = clGetPlatformIDs(num_platforms, platforms, NULL);
  if(ret != CL_SUCCESS) {
    free(platforms);
    return ret;
  }

  size_t size_ret;
  char *_vendor = NULL;
  for(cl_int i = 0; i < num_platforms; i++){
    ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, 0, NULL, &size_ret);
    if(ret != CL_SUCCESS) break;

    _vendor = (char*)realloc((void*)_vendor, size_ret);
    ret = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, size_ret, _vendor, NULL);
    if(ret != CL_SUCCESS) break;

    printf("%s\n", _vendor);
    if(strstr(_vendor, vendor) != NULL) {
      *chosen = platforms[i];
      ret = CL_SUCCESS;
      break;
    }
  }
  
  free(_vendor);
  free(platforms);
  return ret;
}

cl_int init(cl_device_id *devices, cl_context *context, cl_command_queue *queue){
  cl_platform_id platform;
  cl_int err;
  err = get_platform(VENDOR, &platform);
  handle(__LINE__ - 1, errno);

  err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, N_GPUS, devices, NULL);
  handle(__LINE__ - 1, err);

  *context = clCreateContext(NULL, N_GPUS, devices, NULL, NULL, &err);
  handle(__LINE__ - 1, err);

  *queue = clCreateCommandQueue(*context, *devices, 0, &err);
  handle(__LINE__ - 1, err);

  return CL_SUCCESS;
}

void sobel(const unsigned char *in, std::vector<unsigned char> &out,
           unsigned int w, unsigned int h) {
  size_t size = out.size();
  cl_device_id devices;
  cl_context context;
  cl_command_queue queue;
  cl_int err;
  
  init(&devices, &context, &queue);

  cl_mem d_in = clCreateBuffer(context, CL_MEM_READ_ONLY, size, NULL, &err);
  handle(__LINE__ - 1, err);
  cl_mem d_out = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size, NULL, &err);
  handle(__LINE__ - 1, err);

  err = clEnqueueWriteBuffer(queue, d_in, CL_TRUE, 0, size, in, 0, NULL, NULL);
  handle(__LINE__ - 1, err);

  cl_program program = clCreateProgramWithSource(context, 1, &kernel, NULL, &err);
  handle(__LINE__ - 1, err);

  err = clBuildProgram(program, N_GPUS, &devices, NULL, NULL, NULL);
#ifdef DEBUG
  size_t len = 0;
  clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
  char *buffer = (char *)calloc(len, sizeof(char));
  clGetProgramBuildInfo(program, devices, CL_PROGRAM_BUILD_LOG, len, buffer, NULL);
  printf("%s\n", buffer);
  free(buffer);
#endif
  handle(__LINE__ - 1, err);

  cl_kernel kernel = clCreateKernel(program, "find_edge", &err);
  handle(__LINE__ - 1, err);

  err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_in);
  handle(__LINE__ - 1, err);
  err = clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_out);
  handle(__LINE__ - 1, err);
  err = clSetKernelArg(kernel, 2, sizeof(unsigned int), &w);
  handle(__LINE__ - 1, err);
  err = clSetKernelArg(kernel, 3, sizeof(unsigned int), &h);
  handle(__LINE__ - 1, err);

  err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &size, NULL, 0, NULL, NULL);
  handle(__LINE__ - 1, err);

  err = clEnqueueReadBuffer(queue, d_out, CL_TRUE, 0, size, out.data(), 0, NULL, NULL);
  handle(__LINE__ - 1, err);

  clFlush(queue);
  clFinish(queue);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseMemObject(d_out);
  clReleaseMemObject(d_in);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
  clReleaseDevice(devices);
}
