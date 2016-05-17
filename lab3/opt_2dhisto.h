#ifndef OPT_KERNEL
#define OPT_KERNEL

void opt_2dhisto(uint32_t* , size_t , size_t , uint8_t* , uint32_t*);

/* Include below the function headers of any other functions that you implement */

void* AllocateDeviceMemory(size_t);

void CopyToDeviceMemory(void*, void*, size_t );

void CopyFromDeviceMemory(void* , void* , size_t);

void FreeDeviceMemory(void* );

#endif
