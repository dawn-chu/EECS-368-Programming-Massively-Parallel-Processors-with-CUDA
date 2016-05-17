#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

#include <iostream>
#include <cutil_inline.h>
#include <assert.h>


#ifdef ZERO_BANK_CONFLICTS
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS + (index) >> (2*LOG_NUM_BANKS))
#else
#define CONFLICT_FREE_OFFSET(index) ((index) >> LOG_NUM_BANKS)
#endif

#define LOG_NUM_BANKS 5
#define BLOCK_SIZE 256
#define NUM_BANKS 32

__global__ void uniformAdd(float *g_data, float *uniforms, int n, int blockOffset, int baseIndex);

template <bool storeSum, bool isNP2>
__global__ void prescan(float *g_odata, const float *g_idata, float *g_blockSums, int n, int blockIndex, int baseIndex);


void prescanArrayRecursive(float *outArray, const float *inArray, int numElements, int level);

inline int floorPow2(int n);

inline bool isPowerOfTwo(int n);

__device__ void scanFromRootToLeaves(float *s_data, unsigned int stride);


unsigned int numEltsAllocated = 0;
unsigned int numLevelsAllocated = 0;
float** scanBlockSums;

void prescanArray(float *outArray, float *inArray, int numElements)
{
    prescanArrayRecursive(outArray, inArray, numElements, 0);
}


void deallocBlockSums()
{
    for (unsigned int i = 0; i < numLevelsAllocated; i++)
    {
        cudaFree(scanBlockSums[i]);
    }
    free((void**)scanBlockSums);
    scanBlockSums = 0;
    numEltsAllocated = 0;
    numLevelsAllocated = 0;
}


void preallocBlockSums(unsigned int maxNumElements)
{
    assert(numEltsAllocated == 0); // shouldn't be called

    numEltsAllocated = maxNumElements;

    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numElts = maxNumElements;

    int level = 0;

    do
    {
        unsigned int numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            level++;
        }
        numElts = numBlocks;
    } while (numElts > 1);

    scanBlockSums = (float**) malloc(level * sizeof(float*));
    numLevelsAllocated = level;

    numElts = maxNumElements;
    level = 0;

    do
    {
        unsigned int numBlocks =
            max(1, (int)ceil((float)numElts / (2.f * blockSize)));
        if (numBlocks > 1)
        {
            cutilSafeCall(cudaMalloc((void**) &scanBlockSums[level++],
                                      numBlocks * sizeof(float)));
        }
        numElts = numBlocks;
    } while (numElts > 1);
}

inline int floorPow2(int n)
{
    #ifdef WIN32
        return 1 << (int)logb((float)n);
    #else
        int exp;
        frexp((float)n, &exp);
        return 1 << (exp - 1);
    #endif
}

inline bool isPowerOfTwo(int n)
{
    return ((n&(n-1))==0) ;
}


void prescanArrayRecursive(float *outArray, const float *inArray, int numElements, int level){
    unsigned int blockSize = BLOCK_SIZE; // max size of the thread blocks
    unsigned int numBlocks = max(1, (int)ceil((float)numElements / (2.f * blockSize)));
    unsigned int numThreads;

    if (numBlocks > 1)
        numThreads = blockSize;
    else if (isPowerOfTwo(numElements))
        numThreads = numElements / 2;
    else
        numThreads = floorPow2(numElements);

    unsigned int numEltsPerBlock = numThreads * 2;
    unsigned int numEltsLastBlock = numElements - (numBlocks-1) * numEltsPerBlock;
    unsigned int numThreadsLastBlock = max(1, numEltsLastBlock / 2);
    unsigned int np2LastBlock = 0;
    unsigned int sharedMemLastBlock = 0;

    if (numEltsLastBlock != numEltsPerBlock)
    {
        np2LastBlock = 1;
        if(!isPowerOfTwo(numEltsLastBlock))
            numThreadsLastBlock = floorPow2(numEltsLastBlock);
        unsigned int extraSpace = (2 * numThreadsLastBlock) / NUM_BANKS;
        sharedMemLastBlock = sizeof(float) * (2 * numThreadsLastBlock + extraSpace);
    }

    // padding space is used to avoid shared memory bank conflicts
    unsigned int extraSpace = numEltsPerBlock / NUM_BANKS;
    unsigned int sharedMemSize = sizeof(float) * (numEltsPerBlock + extraSpace);

    #ifdef DEBUG
    if (numBlocks > 1)
    {
        assert(numEltsAllocated >= numElements);
    }
    #endif

    dim3  grid(max(1, numBlocks - np2LastBlock), 1, 1);
    dim3  threads(numThreads, 1, 1);

    if (numBlocks > 1)
    {
        prescan<true, false><<< grid, threads, sharedMemSize >>>(outArray, inArray, scanBlockSums[level], numThreads * 2, 0, 0);
        if (np2LastBlock)
        {
            prescan<true, true><<< 1, numThreadsLastBlock, sharedMemLastBlock >>>
                (outArray, inArray, scanBlockSums[level], numEltsLastBlock,
                 numBlocks - 1, numElements - numEltsLastBlock);
        }

        prescanArrayRecursive(scanBlockSums[level], scanBlockSums[level], numBlocks, level+1);
        uniformAdd<<< grid, threads >>>(outArray, scanBlockSums[level], numElements - numEltsLastBlock, 0, 0);
        if (np2LastBlock)
        {
            uniformAdd<<< 1, numThreadsLastBlock >>>(outArray, scanBlockSums[level], numEltsLastBlock, numBlocks - 1, numElements - numEltsLastBlock);
        }
    }
    else if (isPowerOfTwo(numElements))
    {
        prescan<false, false><<< grid, threads, sharedMemSize >>>(outArray, inArray, 0, numThreads * 2, 0, 0);
    }
    else
    {
         prescan<false, true><<< grid, threads, sharedMemSize >>>(outArray, inArray, 0, numElements, 0, 0);
    }
}

template <bool isNP2>
__device__ void loadFromMem(float *s_data, const float *g_idata, int n, int baseIndex, int& ai, int& bi, int& mem_ai, int& mem_bi, int& bankOffsetA, int& bankOffsetB)
{
    int thid = threadIdx.x;
    mem_ai = baseIndex + threadIdx.x;
    mem_bi = mem_ai + blockDim.x;

    ai = thid;
    bi = thid + blockDim.x;

    bankOffsetA = CONFLICT_FREE_OFFSET(ai);
    bankOffsetB = CONFLICT_FREE_OFFSET(bi);

    s_data[ai + bankOffsetA] = g_idata[mem_ai]; 
    
    if (isNP2){
        s_data[bi + bankOffsetB] = (bi < n) ? g_idata[mem_bi] : 0; 
    }
    else{
        s_data[bi + bankOffsetB] = g_idata[mem_bi]; 
    }
}


template <bool isNP2>
__device__ void storeToMem(float* g_odata, const float* s_data, int n, int ai, int bi, int mem_ai, int mem_bi, int bankOffsetA, int bankOffsetB)
{
    __syncthreads();
    g_odata[mem_ai] = s_data[ai + bankOffsetA]; 
    if (isNP2) {
        if (bi < n)
            g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
    else{
        g_odata[mem_bi] = s_data[bi + bankOffsetB]; 
    }
}



template <bool storeSum>
__device__ void clearElements(float* s_data, float *g_blockSums, int blockIndex){
    if (threadIdx.x == 0){
        int index = (blockDim.x << 1) - 1;
        index += CONFLICT_FREE_OFFSET(index);
        if (storeSum) {
            g_blockSums[blockIndex] = s_data[index];
        }
        s_data[index] = 0;
    }
}

__device__ unsigned int getSum(float *s_data){
    unsigned int thid = threadIdx.x;
    unsigned int stride = 1;
    
    for (int d = blockDim.x; d > 0; d >>= 1){
        __syncthreads();

        if (thid < d)      {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            s_data[bi] += s_data[ai];
        }

        stride *= 2;
    }

    return stride;
}

__device__ void scanFromRootToLeaves(float *s_data, unsigned int stride)
{
     unsigned int thid = threadIdx.x;

    for (int d = 1; d <= blockDim.x; d *= 2)
    {
        stride >>= 1;
        __syncthreads();
        if (thid < d)
        {
            int i  = __mul24(__mul24(2, stride), thid);
            int ai = i + stride - 1;
            int bi = ai + stride;

            ai += CONFLICT_FREE_OFFSET(ai);
            bi += CONFLICT_FREE_OFFSET(bi);

            float t  = s_data[ai];
            s_data[ai] = s_data[bi];
            s_data[bi] += t;
        }
    }
}

template <bool storeSum>
__device__ void prescanBlock(float *data, int blockIndex, float *blockSums)
{
    int stride = getSum(data);            
    clearElements<storeSum>(data, blockSums, (blockIndex == 0) ? blockIdx.x : blockIndex);
    scanFromRootToLeaves(data, stride);         
}

__global__ void uniformAdd(float *g_data, float *uniforms, int n, int blockOffset, int baseIndex){
    __shared__ float uni;
    if (threadIdx.x == 0)
        uni = uniforms[blockIdx.x + blockOffset];
    
    unsigned int address = __mul24(blockIdx.x, (blockDim.x << 1)) + baseIndex + threadIdx.x; 

    __syncthreads();
    
    g_data[address]              += uni;
    g_data[address + blockDim.x] += (threadIdx.x + blockDim.x < n) * uni;
}

template <bool storeSum, bool isNP2>
__global__ void prescan(float *g_odata, const float *g_idata, float *g_blockSums, int n, int blockIndex, int baseIndex){
    int ai, bi, mem_ai, mem_bi, bankOffsetA, bankOffsetB;
    extern __shared__ float s_data[];

    loadFromMem<isNP2>(s_data, g_idata, n, (baseIndex == 0) ? 
                                __mul24(blockIdx.x, (blockDim.x << 1)):baseIndex,
                                  ai, bi, mem_ai, mem_bi, 
                                  bankOffsetA, bankOffsetB); 
    prescanBlock<storeSum>(s_data, blockIndex, g_blockSums); 
    storeToMem<isNP2>(g_odata, s_data, n, 
                                 ai, bi, mem_ai, mem_bi, 
                                 bankOffsetA, bankOffsetB);  
}

#endif // _PRESCAN_CU_