/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and
 * international Copyright laws.
 *
 * This software and the information contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a Non-Disclosure Agreement.  Any reproduction or
 * disclosure to any third party without the express written consent of
 * NVIDIA is prohibited.
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 *
 * U.S. Government End Users.  This source code is a "commercial item" as
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of
 * "commercial computer software" and "commercial computer software
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
 * and is provided to the U.S. Government only as a commercial end item.
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	const int Tile_Width = 32;

	__shared__ float sharedM[Tile_Width][Tile_Width+1];
	__shared__ float sharedN[Tile_Width][Tile_Width+1];

	int tx = threadIdx.x;
    int ty = threadIdx.y;
	int bx = blockIdx.x;
	int by = blockIdx.y;

	int row = by* Tile_Width + ty;
	int col = bx* Tile_Width + tx;

	float value =0.0;

	for(int i=0; i< ((P.height+Tile_Width-1)/Tile_Width); i++)
	 {
        if (i*Tile_Width + tx < M.width && row < M.height)
            sharedM[ty][tx] = M.elements[row*M.width + i*Tile_Width + tx];
        else
            sharedM[ty][tx] = 0.0;

        if (i*Tile_Width + ty < N.height && col < N.width)
            sharedN[ty][tx] = N.elements[(i*Tile_Width + ty)*N.width + col];
        else
            sharedN[ty][tx] = 0.0;
        __syncthreads();

        for(int j = 0; j < Tile_Width; j++)
            value += sharedM[ty][j] * sharedN[j][tx];
        __syncthreads();
    }

     if (row < P.height && col < P.width)
        P.elements[row*P.width + col] = value;


}


#endif
