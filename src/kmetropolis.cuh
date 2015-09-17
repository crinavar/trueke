//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  trueke                                                                      //
//  A multi-GPU implementation of the exchange Monte Carlo method.              //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of trueke.                                                //
//  trueke is free software: you can redistribute it and/or modify              //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  trueke is distributed in the hope that it will be useful,                   //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with trueke.  If not, see <http://www.gnu.org/licenses/>.             //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////

#ifndef _KERNEL_MONTECARLO_CUH_
#define _KERNEL_MONTECARLO_CUH_

// GPU-related definitions

#define sLx (BX+2)
#define sLy (BY+2)
#define sLz (BZ+2)

#define SVOLUME ((sLx)*(sLy)*(sLz))
#define BVOLUME ((BX)*((BY)/2)*(BZ))

#define MAX_BLOCK_VOLUME 1024
#define BLOCKSIZE 8
#define BLOCKSIZE1D 512
#define BLOCK_STEPS 1
#define EPSILON 0.00000000001f

#define NORM 4.656612e-10f
#define AFACTOR 1664525
#define BFACTOR 1013904223
#define RAN(n) (n = AFACTOR*n + BFACTOR)

#define C(x,y,z,L) 		((z)*(L)*(L)+(y)*(L)+(x))
#define sC(x,y,z,Lx,Ly)	((z+1)*(Ly)*(Lx)+(y+1)*(Lx)+(x+1))

#define PI 3.141592653589793238f

typedef int site_t;
// kernel_metropolis
__global__ void
kernel_metropolis(int N, int L, site_t *s, int *H, float h, float B, curandState *dstates, int alt){

	// offsets
	int offx = blockIdx.x * BX;
	int offy = (2*blockIdx.y + ((blockIdx.x + blockIdx.z + alt) & 1)) * BY;
	int offz = blockIdx.z * BZ;

	// halo shared memory coords
	int sx = threadIdx.x;
	int sy1 = 2*threadIdx.y;
	int sy2 = 2*threadIdx.y + 1;
	int sz = threadIdx.z;

	// global coords
	int x = offx + sx;
	int y1 = offy + sy1;
	int y2 = offy + sy2;
	int z = offz + sz;

	//if(x >= N || y1 >= N || y2 >= N || z >= N)
		//return;

	// global and local and block id in soc
	int tid = z*L*L/4 + (blockIdx.y * BY/2 + threadIdx.y)*L + x;
	// shared memory
	__shared__ site_t ss[SVOLUME];

	// load the spins into shared memory
	ss[sC(sx, sy1, sz, sLx, sLy)] = s[C(x, y1, z, L)];
	ss[sC(sx, sy2, sz, sLx, sLy)] = s[C(x, y2, z, L)];
	// get the h1,h2 values for y1 y2.
	int h1 = H[C(x, y1, z, L)];
	int h2 = H[C(x, y2, z, L)];

	// ------------------------------------------------
	// halo
	// ------------------------------------------------
	// Y boundary
	if(threadIdx.y == 0){
		// we also check if we are on the limit of the lattice
		ss[sC(sx, -1, sz, sLx, sLy)] = (offy == 0) ? s[C(x, L-1, z, L)] : s[C(x, offy-1, z, L)];
	}
	if(threadIdx.y == BY/2-1){
		ss[sC(sx, BY, sz, sLx, sLy)] = (offy == L-BY) ? s[C(x, 0, z, L)] : s[C(x, offy+BY, z, L)];
	}

	// X boundary
	if(threadIdx.x == 0){
		if(blockIdx.x == 0){
			ss[sC(-1, sy1, sz, sLx, sLy)] = s[C(L-1, y1, z, L)];
			ss[sC(-1, sy2, sz, sLx, sLy)] = s[C(L-1, y2, z, L)];
		}
		else{
			ss[sC(-1, sy1, sz, sLx, sLy)] = s[C(offx-1, y1, z, L)];
			ss[sC(-1, sy2, sz, sLx, sLy)] = s[C(offx-1, y2, z, L)];
		}
	}
	if(threadIdx.x == BX-1){
		if(blockIdx.x == gridDim.x-1){
			ss[sC(BX, sy1, sz, sLx, sLy)] = s[C(0, y1, z, L)];
			ss[sC(BX, sy2, sz, sLx, sLy)] = s[C(0, y2, z, L)];
		}
		else{
			ss[sC(BX, sy1, sz, sLx, sLy)] = s[C(offx+BX, y1, z, L)];
			ss[sC(BX, sy2, sz, sLx, sLy)] = s[C(offx+BX, y2, z, L)];
		}
	}

	// Z boundary
	if(threadIdx.z == 0){
		if(blockIdx.z == 0){
			ss[sC(sx, sy1, -1, sLx, sLy)] = s[C(x, y1, L-1, L)];
			ss[sC(sx, sy2, -1, sLx, sLy)] = s[C(x, y2, L-1, L)];
		}
		else{
			ss[sC(sx, sy1, -1, sLx, sLy)] = s[C(x, y1, offz-1, L)];
			ss[sC(sx, sy2, -1, sLx, sLy)] = s[C(x, y2, offz-1, L)];
		}
	}
	if(threadIdx.z == BZ-1){
		if(blockIdx.z == gridDim.z-1){
			ss[sC(sx, sy1, BZ, sLx, sLy)] = s[C(x, y1, 0, L)];
			ss[sC(sx, sy2, BZ, sLx, sLy)] = s[C(x, y2, 0, L)];
		}
		else{
			ss[sC(sx, sy1, BZ, sLx, sLy)] = s[C(x, y1, offz+BZ, L)];
			ss[sC(sx, sy2, BZ, sLx, sLy)] = s[C(x, y2, offz+BZ, L)];
		}
	}

	// get random number state
	curandState lstate= dstates[tid];
	// the white and black y
  	int wy = ((sx + sz) & 1) 		+ 2*threadIdx.y;
  	int by = ((sx + sz + 1) & 1)	+ 2*threadIdx.y;
	float dh;
	int c;

	__syncthreads();
	//#pragma unroll
	for(int i = 0; i < BLOCK_STEPS; ++i){

		/* -------- white update -------- */
		dh = (float)(ss[sC(sx, wy, sz, sLx, sLy)] * (	(float)(ss[sC(sx-1,wy,sz, sLx, sLy)] + ss[sC(sx+1, wy, sz, sLx, sLy)]	+ ss[sC(sx,wy-1,sz, sLx, sLy)] + ss[sC(sx, wy+1, sz, sLx, sLy)]	+
																ss[sC(sx,wy,sz-1, sLx, sLy)] + ss[sC(sx, wy, sz+1, sLx, sLy)])	+ h*h1));
		c = signbit(dh-EPSILON) | signbit(curand_uniform(&lstate) - expf(dh * B));
		ss[sC(sx, wy, sz, sLx, sLy)] *= (1 - 2*c);
		__syncthreads();

		/* -------- black update -------- */
		dh = (float)(ss[sC(sx, by, sz, sLx, sLy)] * (	(float)(ss[sC(sx-1,by,sz, sLx, sLy)] + ss[sC(sx+1, by, sz, sLx, sLy)]	+ ss[sC(sx,by-1,sz, sLx, sLy)] + ss[sC(sx, by+1, sz, sLx, sLy)]	+
																ss[sC(sx,by,sz-1, sLx, sLy)] + ss[sC(sx, by, sz+1, sLx, sLy)])	+ h*h2));

		c = signbit(dh-EPSILON) | signbit(curand_uniform(&lstate) - expf(dh * B));
		ss[sC(sx, by, sz, sLx, sLy)] *= (1 - 2*c);
		__syncthreads();
	}

	/* copy data back to gmem */
	s[C(x, y1, z, L)] = ss[sC(sx, sy1, sz, sLx, sLy)];
	s[C(x, y2, z, L)] = ss[sC(sx, sy2, sz, sLx, sLy)]; 
	/* update random number state */
	dstates[tid] = lstate;
}

// kernel for random init
// WARNING: the space of computation is 1/4 of N, so that is why each thread does quadruple work.
__global__ void kernel_reset_random(int *s, int N, curandState *state){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	float v;
	/* Each thread gets same seed, a different sequence number, no offset */
	if( x >= N/4 )
		return;

	/* get the prng state in register memory */
	curandState lstate = state[x];

	// first random
	v = (int) (curand_uniform(&lstate) + 0.5f);
	s[x] = 1-2*(int)v;
	// second random
	v = (int) (curand_uniform(&lstate) + 0.5f);
	s[x + N/4] = 1-2*(int)v;
	// third random
	v = (int) (curand_uniform(&lstate) + 0.5f);
	s[x + N/2 ]	= 1-2*(int)v;
	// fourth random
	v = (int) (curand_uniform(&lstate) + 0.5f);
	s[x + 3*N/4] = 1-2*(int)v;

	/* save the state back to global memory */
	state[x] = lstate;
}

template<typename T>
__global__ void kernel_reset(T *a, int N, T val){
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < N)
		a[idx] = val;
}
#endif
