# How to compile?

Since this is sobel implementation in OpenCL, you need to setup some dependencies first.
In palmetto, you need to request a node that has a GPU:

```sh
qsub -I -l select=1:ncpus=16:mem=14gb:ngpus=1:gpu_model=k20,walltime=24:00:00
```

Then, load cuda-toolkit so that the program can utilize the GPU:

```sh
module load cuda-toolkit
```

Go into the source code directory and issue:

```sh
make
```

# How to run?

```
./sobel <image_path>
```

# Notes

This program doesn't have the third argument to specify the number of threads. This is because
having that as a parameter is not as easy as doing that in the CPU implementation. For example,
OpenCL will not allow the kernel to run if the local_work_size (thread size in CPU) doesn't divide
the global_work_size (the image size). Meaning, during runtime, the program has to somehow get the
nearest divisible number depending the image size.  In the meantime, this dynamicallly adjusting
the number of threads will change the variable itself anyway and this adds another reason why I 
didn't provide the third argument.
