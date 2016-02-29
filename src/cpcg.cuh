/*
 * PCG Random Number Generation for C.
 *
 * Copyright 2014 Melissa O'Neill <oneill@pcg-random.org>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * For additional information about the PCG random number generation scheme,
 * including its license and other licensing options, visit
 *
 *       http://www.pcg-random.org
 */

/*
 * This code is derived from the full C implementation, which is in turn
 * derived from the canonical C++ PCG implementation. The C++ version
 * has many additional features and is preferable if you can use C++ in
 * your project.
 */

/* simple GPU-based PCG by Cristobal A. Navarro, Feb 9, 2016 */
#ifndef CPCG_H
#define CPCG_H

#define INV_UINT_MAX 2.3283064e-10f

#include <limits.h>
#include <inttypes.h>
struct __align__(16) pcg_state_setseq_64 {    // Internals are *Private*.
    uint64_t state;             // RNG state.  All values are possible.
    uint64_t inc;               // Controls which RNG sequence (stream) is // selected. Must *always* be odd.
};

typedef struct pcg_state_setseq_64 pcg32_random_t;
__device__ inline uint32_t pcg32_random_r(pcg32_random_t* rng);
__device__ inline uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc);

// gpu 16byte aligned struct version
__device__ inline void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq){
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    pcg32_random_r(rng);
    rng->state += initstate;
    pcg32_random_r(rng);
}

__device__ inline uint32_t pcg32_random_r(pcg32_random_t* rng){
    uint64_t oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

__device__ inline float pcgrand(pcg32_random_t *rng){
    return (float) pcg32_random_r(rng) / (float)(UINT_MAX);
}


// fully separated GPU version
__device__ inline void gpu_pcg32_srandom_r(uint64_t *state, uint64_t *inc, uint64_t initstate, uint64_t initseq){
    *state = 0U;
    *inc = (initseq << 1u) | 1u;
    gpu_pcg32_random_r(state, inc);
    *state += initstate;
    gpu_pcg32_random_r(state, inc);
}


__device__ inline uint32_t gpu_pcg32_random_r(uint64_t *state, uint64_t *inc){
    uint64_t oldstate = *state;
    *state = oldstate * 6364136223846793005ULL + *inc;
    uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
    //return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31)))*INV_UINT_MAX;
}

__device__ inline float gpu_rand01(uint64_t *state, uint64_t *inc){
    //return (float) gpu_pcg32_random_r(state, inc) / (float)(UINT_MAX);
    return (float) gpu_pcg32_random_r(state, inc) * INV_UINT_MAX;
}

#endif

