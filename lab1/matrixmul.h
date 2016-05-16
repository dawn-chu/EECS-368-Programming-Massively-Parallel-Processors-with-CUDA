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

#ifndef _MATRIXMUL_H_
#define _MATRIXMUL_H_

// Thread block size
#define MATRIX_SIZE 16

// Matrix dimensions
// (chosen as multiples of the thread block size for simplicity)
#define WM MATRIX_SIZE // Matrix M width
#define HM MATRIX_SIZE // Matrix M height
#define WN MATRIX_SIZE // Matrix N width
#define HN WM  // Matrix N height
#define WP WN  // Matrix P width 
#define HP HM  // Matrix P height


// Matrix Structure declaration
typedef struct {
	//width of the matrix represented
    unsigned int width;
	//height of the matrix represented
    unsigned int height;
	//number of elements between the beginnings of adjacent
	// rows in the memory layout (useful for representing sub-matrices)
    unsigned int pitch;
	//Pointer to the first element of the matrix represented
    float* elements;
} Matrix;


#endif // _MATRIXMUL_H_

