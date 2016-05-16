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
 * Host code.
 */

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, project
#include <cutil.h>

// includes, kernels
#include <matrixmul_kernel.cu>

////////////////////////////////////////////////////////////////////////////////
// declarations, forward

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);

Matrix AllocateDeviceMatrix(const Matrix M);
Matrix AllocateMatrix(int height, int width, int init);
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost);
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice);
int ReadFile(Matrix* M, char* file_name);
void WriteFile(Matrix M, char* file_name);

void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P);


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) {

	// Matrices for the program
	Matrix  M;
	Matrix  N;
	Matrix  P;
	// Number of elements in the solution matrix
	//  Assuming square matrices, so the sizes of M, N and P are equal
	unsigned int size_elements = WP * HP;
	int errorM = 0, errorN = 0;

	srand(2012);

	// Check command line for input matrix files
	if(argc != 3 && argc != 4) 
	{
		// No inputs provided
		// Allocate and initialize the matrices
		M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
		N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 1);
		P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	}
	else
	{
		// Inputs provided
		// Allocate and read source matrices from disk
		M  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
		N  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);		
		P  = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
		errorM = ReadFile(&M, argv[1]);
		errorN = ReadFile(&N, argv[2]);
		// check for read errors
		if(errorM != size_elements || errorN != size_elements)
		{
			printf("Error reading input files %d, %d\n", errorM, errorN);
			return 1;
		}
	}

	// M * N on the device
	MatrixMulOnDevice(M, N, P);

	// compute the matrix multiplication on the CPU for comparison
	Matrix reference = AllocateMatrix(MATRIX_SIZE, MATRIX_SIZE, 0);
	computeGold(reference.elements, M.elements, N.elements, HM, WM, WN);

	// check if the device result is equivalent to the expected solution
	CUTBoolean res = cutComparefe(reference.elements, P.elements, size_elements, 0.0001f);
	printf("Test %s\n", (1 == res) ? "PASSED" : "FAILED");

	// output result if output file is requested
	if(argc == 4)
	{
		WriteFile(P, argv[3]);
	}
	else if(argc == 2)
	{
		WriteFile(P, argv[1]);
	}    

	// Free host matrices
	free(M.elements);
	M.elements = NULL;
	free(N.elements);
	N.elements = NULL;
	free(P.elements);
	P.elements = NULL;
	return 0;
}


////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for C///////////////////////////////////////////////////////////////////////////////
void MatrixMulOnDevice(const Matrix M, const Matrix N, Matrix P)
{	//Interface host call to the device kernel code and invoke the kernel
	
	
	Matrix  d_M = AllocateDeviceMatrix(M);
	Matrix	d_N = AllocateDeviceMatrix(N);
	Matrix	d_P = AllocateDeviceMatrix(P);

  CopyToDeviceMatrix(d_M, M);
  CopyToDeviceMatrix(d_N, N);

  dim3 dimBlock(MATRIX_SIZE, MATRIX_SIZE);
  dim3 dimGrid(N.width / dimBlock.x, M.height / dimBlock.y);
  MatrixMulKernel<<<dimGrid, dimBlock>>>(d_M, d_N, d_P);

  CopyFromDeviceMatrix(P, d_P);
//  printf("P.elements[0]: %f", P.elements[0]);

  cudaFree(d_M.elements);
  cudaFree(d_N.elements);
  cudaFree(d_P.elements);	
	
	
	
}

// Allocate a device matrix of same size as M.
Matrix AllocateDeviceMatrix(const Matrix M)
{
	Matrix Mdevice = M;
	int size = M.width * M.height * sizeof(float);
	cudaMalloc((void**)&Mdevice.elements, size);
	return Mdevice;
}

// Allocate a matrix of dimensions height*width
//	If init == 0, initialize to all zeroes.  
//	If init == 1, perform random initialization.
Matrix AllocateMatrix(int height, int width, int init)
{
	Matrix M;
	M.width = M.pitch = width;
	M.height = height;
	int size = M.width * M.height;
	M.elements = NULL;

	M.elements = (float*) malloc(size*sizeof(float));

	for(unsigned int i = 0; i < M.height * M.width; i++)
	{
		M.elements[i] = (init == 0) ? (0.0f) : (rand() / (float)RAND_MAX);
	}
	return M;
}	

// Copy a host matrix to a device matrix.
void CopyToDeviceMatrix(Matrix Mdevice, const Matrix Mhost)
{
	int size = Mhost.width * Mhost.height * sizeof(float);
	Mdevice.height = Mhost.height;
	Mdevice.width = Mhost.width;
	Mdevice.pitch = Mhost.pitch;
	cudaMemcpy(Mdevice.elements, Mhost.elements, size, 
			cudaMemcpyHostToDevice);
}

// Copy a device matrix to a host matrix.
void CopyFromDeviceMatrix(Matrix Mhost, const Matrix Mdevice)
{
	int size = Mdevice.width * Mdevice.height * sizeof(float);
	cudaMemcpy(Mhost.elements, Mdevice.elements, size, 
			cudaMemcpyDeviceToHost);
}

// Read a 16x16 floating point matrix in from file
int ReadFile(Matrix* M, char* file_name)
{
	unsigned int data_read = MATRIX_SIZE*MATRIX_SIZE;
	cutReadFilef(file_name, &(M->elements), &data_read, true);
	return data_read;
}

// Write a 16x16 floating point matrix to file
void WriteFile(Matrix M, char* file_name)
{
	cutWriteFilef(file_name, M.elements, M.width*M.height, 0.0001f);
}
