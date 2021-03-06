#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 25
__global__ void mandelKernel(
        int* d_img, float lowerX, float lowerY, float stepX,
        float stepY, int width, int height, int maxIterations, 
        int g_width, int g_height, size_t pitch) {
    // To avoid error caused by the floating number, use the following pseudo code
    // float x = lowerX + thisX * stepX;
    // float y = lowerY + thisY * stepY;

    unsigned int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    int end_j = thisY + g_height;
    int end_i = thisX + g_width;
    int * rowhead;
    for (int j = thisY; j < end_j; j++)
    {
        rowhead = (int*)((char*)d_img + j*pitch);
        for (int i = thisX; i < end_i; i++)
        {
            if (i < width && j < height) {
                // int idx = j * width + i;
                float c_re = lowerX + i * stepX;
                float c_im = lowerY + j * stepY;
                float z_re = c_re, z_im = c_im;
                int val = 0;
                for (val = 0; val < maxIterations; val++)
                {
                    if (z_re * z_re + z_im * z_im > 4.f)
                        break;
                    float new_re = z_re * z_re - z_im * z_im;
                    float new_im = 2.f * z_re * z_im;
                    z_re = c_re + new_re;
                    z_im = c_im + new_im;
                }
                rowhead[i] = val;
            }
        }
    }
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    int* d_img, *host_img;
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    // cudaMalloc((void **)&d_img, resX * resY * sizeof(int)); // kernel1
    // host_img = (int *) malloc(resX * resY * sizeof(int)); // kernel1
    size_t pitch; // kernel2
    cudaMallocPitch((void **)&d_img, &pitch, sizeof(float)*resX, resY); // kernel2
    cudaHostAlloc((void **)&host_img, resX * resY * sizeof(int),cudaHostAllocDefault); // kernel2
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    int g_width = numBlock.x / BLOCK_SIZE;
    int g_height = numBlock.y / BLOCK_SIZE;
    printf("%d\n", pitch);
    mandelKernel<<<numBlock, blockSize>>>(d_img, lowerX, lowerY, stepX, stepY, resX, resY, maxIterations, g_width, g_height, pitch);
    cudaDeviceSynchronize();
    cudaMemcpy2D(host_img, sizeof(int)*resX, d_img, pitch, sizeof(int)*resX, pitch, cudaMemcpyDeviceToHost);
    memcpy(img, host_img,resX * resY * sizeof(int));
    cudaFree(d_img);
    free(host_img);
}
