#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#define BLOCK_SIZE 16
__global__ void mandelKernel(int* d_img, float lowerX, float lowerY, float stepX, float stepY, int width, int height, int maxIterations) 
{
    // To avoid error caused by the floating number, use the following pseudo code
    //

    unsigned int thisX = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int thisY = blockIdx.y * blockDim.y + threadIdx.y;
    printf("blockIdx.x:%d, threadIdx.x:%d, blockIdx.y:%d, threadIdx.y:%d thisX:%d, thisY:%d\n",blockIdx.x,threadIdx.x,blockIdx.y,threadIdx.y,thisX, thisY);
    // printf("blockIdx.y:%d, blockDim.y:%d, threadIdx.y:%d\n", blockIdx.y, blockDim.y, threadIdx.y);
    float z_re = lowerX + thisX * stepX;
    float z_im = lowerY + thisY * stepY;

    int idx = thisY * width + thisX;
    // printf("idx:%d\n", idx);
    int i = 0;

    while(z_re * z_re + z_im * z_im <= 4 && i < maxIterations) {
        float new_re = z_re * z_re - z_im * z_im;
        float new_im = 2.f * z_re * z_im;
        z_re = thisX + new_re;
        z_im = thisY + new_im;
        i++;
    }

    // for (i = 0; i < maxIterations; ++i)
    // {
    //     if (z_re * z_re + z_im * z_im > 4.f)
    //         break;
    //     float new_re = z_re * z_re - z_im * z_im;
    //     float new_im = 2.f * z_re * z_im;
    //     z_re = thisX + new_re;
    //     z_im = thisY + new_im;
    // }
    d_img[idx] = i;
}

// Host front-end function that allocates the memory and launches the GPU kernel
void hostFE (float upperX, float upperY, float lowerX, float lowerY, int* img, int resX, int resY, int maxIterations)
{
    
    int* d_img;
    float stepX = (upperX - lowerX) / resX;
    float stepY = (upperY - lowerY) / resY;
    cudaMalloc((void **)&d_img, resX * resY * sizeof(float));
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlock(resX / BLOCK_SIZE, resY / BLOCK_SIZE);
    // printf("Grid : {%d, %d, %d} blocks. Blocks : {%d, %d, %d} threads.\n", gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z);
    mandelKernel<<<numBlock, blockSize>>>(d_img, lowerX, lowerY, stepX, stepY, resX, resY, maxIterations);
    cudaDeviceSynchronize();
    cudaMemcpy(img, d_img, resX * resY * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_img);
}
