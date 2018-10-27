#include <vector>
#include "sobel.h"
#include <stdio.h>
#include <string.h>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

const char *kernel = 
"__constant sampler_t sampler =\n"
"      CLK_NORMALIZED_COORDS_FALSE\n"
"    | CLK_ADDRESS_CLAMP_TO_EDGE\n"
"    | CLK_FILTER_NEAREST;\n"
"void __kernel find_edge(__read_only image2d_t in, __write_only image2d_t out) {\n"
"const int2 pos = {get_global_id(1), get_global_id(0)};\n"
"// Compute gradient in +ve x direction\n"
"float4 gradient_X = read_imagef(in, sampler, pos + (int2)(-1, -1))\n"
"- read_imagef(in, sampler, pos + (int2)(-1, 1))\n"
"+ 2 * read_imagef(in, sampler, pos + (int2)(0, -1))\n"
"- 2 * read_imagef(in, sampler, pos + (int2)(0, 1))\n"
"+ read_imagef(in, sampler, pos + (int2)(1, -1))\n"
"- read_imagef(in, sampler, pos + (int2)(1, 1));\n"
"// Compute gradient in +ve y direction\n"
"float4 gradient_Y = read_imagef(in, sampler, pos + (int2)(-1, -1))\n"
"+ 2 * read_imagef(in, sampler, pos + (int2)(-1, 0))\n"
"+ read_imagef(in, sampler, pos + (int2)(-1, 1))\n"
"- read_imagef(in, sampler, pos + (int2)(1, -1))\n"
"- 2 * read_imagef(in, sampler, pos + (int2)(1, 0))\n"
"- read_imagef(in, sampler, pos + (int2)(1, 1));\n"
"float4 value = (float4)1.0 - sqrt(pow(gradient_X, 2) + pow(gradient_Y, 2));\n"
"write_imagef(out, pos, value);\n"
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

    printf("Found %s!\n", _vendor);
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
  handle(__LINE__ - 1, err);

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
  size_t size = w * h;
  size_t offset[] = {1, 1};
  size_t global_work_size[] = {h - 1, w - 1};
  cl_device_id devices;
  cl_context context;
  cl_command_queue queue;
  cl_int err;
  
  init(&devices, &context, &queue);

  const cl_image_format format = {CL_INTENSITY, CL_UNORM_INT8};
  cl_mem d_in = clCreateImage2D(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format,
                        w, h, 0, (void*)in, &err);
  handle(__LINE__ - 1, err);

  cl_mem d_out = clCreateImage2D(context, CL_MEM_WRITE_ONLY, &format,
                        w, h, 0, NULL, &err);
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

  err = clEnqueueNDRangeKernel(queue, kernel, 2, offset, global_work_size, NULL, 0, NULL, NULL);
  handle(__LINE__ - 1, err);

  const size_t origin[] = {0, 0, 0};
  const size_t region[] = {w, h, 1};
  err = clEnqueueReadImage(queue, d_out, CL_TRUE, origin, region, 0, 0, out.data(), 0, NULL, NULL);
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