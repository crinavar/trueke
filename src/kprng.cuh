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
#ifndef _KERNEL_PRNG_SETUP_
#define _KERNEL_PRNG_SETUP_

__global__ void kernel_prng_setup(curandState *state, int N, unsigned long long seed, unsigned long long seq){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	/* Each thread gets same seed, a different sequence number, no offset */
	if( x < N ){
		/* medium quality, faster */
		curand_init(seed + (unsigned long long)x, seq, 0ULL, &state[x]);
		//curand_init(0, 0, 0, &state[x]);
		/* high quality, slower */
		//curand_init(seed, x, 0, &state[x]);
	}
}

__global__ void kernel_gpupcg_setup(uint64_t *state, uint64_t *inc, int N, int seed, unsigned long long seq){
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if( x < N ){
        gpu_pcg32_srandom_r(&(state[x]), &(inc[x]), x + seed, seq);
	}
}
#endif
