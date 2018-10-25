#pragma once

#define N_GPUS 1
#define VENDOR "NVIDIA"
#define N_THREADS 512

void sobel(const unsigned char *in, std::vector<unsigned char> &out,
          unsigned int w, unsigned int h);