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
#ifndef _SETUP_H_
#define _SETUP_H_

/* function declarations */
void pickgpus(setup_t *s);
void init(setup_t *s, int argc, int argv);
void adapt_init(setup_t *s, int argc, char argv);
void printparams(setup_t *s);
void getparams(setup_t *s, int argc, char **argv);
void newseed(int *seed);
void malloc_arrays(setup_t *s);
void adapt_malloc_arrays(setup_t *s);
void reset(setup_t *s);
void adjustparams(setup_t *s);

/* adapt init */
void adapt_init(setup_t *s, int argc, char **argv){
	printf("adapt_init....{\n");
	fflush (stdout);
	/* get parameters */
	getparams(s, argc, argv);
	/* adjust some parameters related to memory pool and active replicas*/
	adjustparams(s);
    #ifdef MEASURE
        /* folders for output */
        s->obsfolder = "data";
        s->plotfolder = "plots";
        make_output_folders(s->obsfolder, s->plotfolder);
    #endif
	/* random seeds */
	//s->seed = time(NULL);
	//s->nseed = s->seed + 7919;
    /* constant seed useful for debugging */
    //s->seed = s->nseed = 7919;
    s->seed = s->nseed = (unsigned long long)time(NULL);
	srand(s->nseed);
	/* pick the GPUs */
	pickgpus(s);
	/* set the number of threads as the number of GPUs */
	omp_set_num_threads(s->ngpus);
	/* build the space of computation for the lattices */
	s->mcblock = dim3(BX, BY / 2, BZ);
	s->mcgrid = dim3((s->L + BX - 1) / BX, (s->L + BY - 1) / (2 * BY),
			(s->L + BZ - 1) / BZ);
	s->lblock = dim3(BLOCKSIZE1D, 1, 1);
	s->lgrid = dim3((s->N + BLOCKSIZE1D - 1) / BLOCKSIZE1D, 1, 1);

	/* build the space of computation for random numbers and lattice simulation */
	s->prng_block = dim3(BLOCKSIZE1D, 1, 1);
	s->prng_grid = dim3(((s->N / 4) + BLOCKSIZE1D - 1) / BLOCKSIZE1D, 1, 1);

	/* allocate main arrays */
	adapt_malloc_arrays(s);

	/* create timers */
	sdkCreateTimer(&(s->timer));
	sdkCreateTimer(&(s->gtimer));
	sdkCreateTimer(&(s->ktimer));

	/* reset timers */
	sdkResetTimer(&(s->timer));
	sdkResetTimer(&(s->gtimer));
	sdkResetTimer(&(s->ktimer));

	/* print parameters */
	printparams(s);
	printf("}:ok\n\n");
	fflush(stdout);
}

/* adapt malloc */
void adapt_malloc_arrays( setup_t *s ){
	/* multi-gpu adaptation arrays */
	s->mdlat = (int***) malloc(sizeof(int**) * s->ngpus);
	s->aex = (float**)malloc(sizeof(float*)*s->ngpus);
	s->aavex = (float**)malloc(sizeof(float*)*s->ngpus);
	s->aexE = (float**)malloc(sizeof(float*)*s->ngpus);
	s->arstream = (cudaStream_t**)malloc(sizeof(cudaStream_t*) * s->ngpus);
	s->apcga = (uint64_t***)malloc(sizeof(uint64_t**) * s->ngpus);
	s->apcgb = (uint64_t***)malloc(sizeof(uint64_t**) * s->ngpus);
	s->dH = (int **)malloc(sizeof(int*) * s->ngpus);
	s->dE = (float**)malloc(sizeof(float*) * s->ngpus);
	s->arts = (findex_t**)malloc(sizeof(findex_t*) * s->ngpus);
	s->atrs = (findex_t**)malloc(sizeof(findex_t*) * s->ngpus);
	s->aT = (float**)malloc(sizeof(float*)*s->ngpus);

	/* T is a sorted temp array */
	s->T = (float*)malloc(sizeof(float)*s->Ra);
	/* host values for each replica */
	s->E = (float*)malloc(sizeof(float)*s->Ra);
	// memory for H array
	s->hH = (int*)malloc(sizeof(int) * s->N);

	/* multi-GPU setup */
	#pragma omp parallel
	{
		int tid, nt, r;
		/* set threads */
		adapt_threadset(s, &tid, &nt, &r);
		//printf("arge malloc:  tid=%i    r=%i   rpool = %i\n", tid, r, s->rpool[tid]); fflush(stdout);
        /* allocate the replica pool for each GPU */
        s->mdlat[tid] = (int**) malloc(sizeof(int *) * s->rpool[tid]);
    	/* ex is a per temperature counter array */
    	s->aex[tid] = (float*)malloc(sizeof(float)*s->rpool[tid]);
    	/* avex is a per temperature counter array */
    	s->aavex[tid] = (float*)malloc(sizeof(float)*s->rpool[tid]);
    	/* exchange energies */
    	s->aexE[tid] = (float*)malloc(sizeof(float) * s->rpool[tid]);
    	/* CUDA streams */
		s->arstream[tid] = (cudaStream_t*)malloc(sizeof(cudaStream_t) * s->rpool[tid]);
		/* PRNG states volume, one state per thread */
		s->apcga[tid] = (uint64_t**)malloc(sizeof(uint64_t*) * s->rpool[tid]);
		s->apcgb[tid] = (uint64_t**)malloc(sizeof(uint64_t*) * s->rpool[tid]);
		/* fragmented indices for replicas temperature sorted */
		s->arts[tid] = (findex_t*)malloc(sizeof(findex_t)*s->rpool[tid]);
		/* fragmented indices for temperatures replica sorted */
		s->atrs[tid] = (findex_t*)malloc(sizeof(findex_t)*s->rpool[tid]);
		/* fragmented temperatures sorted */
		s->aT[tid] = (float*)malloc(sizeof(float)*s->rpool[tid]);
		/* malloc device magnetic field -- multi-GPU */
		checkCudaErrors(cudaMalloc(&(s->dH[tid]), sizeof(int)*s->N));
		/* malloc device energy reductions -- multi-GPU*/
		checkCudaErrors(cudaMalloc(&(s->dE[tid]), sizeof(float)*s->rpool[tid]));
		/* malloc the data for 'r' replicas on each GPU */
		for(int k = 0; k < s->rpool[tid]; ++k){
			checkCudaErrors(cudaMalloc(&(s->mdlat[tid][k]), sizeof(int) * s->N));
			checkCudaErrors(cudaMalloc(&(s->apcga[tid][k]), (s->N/4) * sizeof(uint64_t)));
			checkCudaErrors(cudaMalloc(&(s->apcgb[tid][k]), (s->N/4) * sizeof(uint64_t)));
			checkCudaErrors(cudaStreamCreateWithFlags(&(s->arstream[tid][k]), cudaStreamNonBlocking));
			kernel_gpupcg_setup<<<s->prng_grid, s->prng_block, 0, s->arstream[tid][k] >>>(s->apcga[tid][k], s->apcgb[tid][k], s->N/4, s->seed, (unsigned long long)(SEQOFFSET*tid + k));
			cudaCheckErrors("kernel: prng reset");
		}
	}
	/* host memory setup for each replica */
	for(int i = 0; i < s->R; i++){
		/* array of temperatures increasing order */
		s->T[i] = s->TR - (s->R-1 - i)*s->dT;
	}
	int count = 0;
	for(int k = 0; k < s->ngpus; ++k){
		for(int j = 0; j < s->gpur[k]; ++j){
			s->arts[k][j] = s->atrs[k][j] = (findex_t){k, j};
			s->aT[k][j] = s->TR - (float)(s->R-1 - count)*s->dT;
			s->aex[k][j] = 0;
			++count;
		}
	}
}

/* set parameters */
void adjustparams(setup_t *s){
    /* total number of spins per replica */
    s->N = (s->L)*(s->L)*(s->L);
    /* shared memory steps */
    s->cs = BLOCK_STEPS;
    /* adjust R to a multiple of ngpus; R' = ceil(R/ngpus) *ngpus */
    s->R = (int)ceil((float)s->R/(float)s->ngpus) * s->ngpus;
    /* compute Ra to be the final size Ra = R + TL */
    s->Ra = s->R + (s->atrials * s->ains);
    /* set replica pools for each GPU */
    s->gpur = (int*)malloc(sizeof(int) * s->ngpus);
    s->rpool = (int*)malloc(sizeof(int) * s->ngpus);
    /* measure zone */
    if( s->mzone == -1 ){
        s->mzone = (int) ((double)s->pts / log2(2.0 + sqrtf((double)s->pts)/(double)s->L) );
    }
    /* last adaptation insert */
    s->fam = 0;

    for(int i=0; i < s->ngpus; ++i){
        /* active replicas per gpu */
        s->gpur[i] = s->R / s->ngpus;
        //printf("s->gpur[%i] = %i\n", i, s->gpur[i]); fflush(stdout); getchar();
        /* replica pool per gpu */
        s->rpool[i] = s->Ra / s->ngpus;
        /* place the remainder of replicas  */
        if( i < (s->Ra % s->ngpus) ){
            s->rpool[i] += 1;
        }
    }
}

/* init */
void init(setup_t *s, int argc, char **argv){
	/* set the number of threads as the number of GPUs */
	//omp_set_num_threads(s->ngpus);
	/* build the space of computation for the lattices */
	s->mcblock = dim3(BX, BY/2, BZ);
	s->mcgrid = dim3((s->L + BX - 1)/BX, (s->L + BY - 1)/(2*BY),  (s->L + BZ - 1)/BZ);
	s->lblock = dim3( BLOCKSIZE1D, 1, 1);
	s->lgrid = dim3((s->N + BLOCKSIZE1D - 1)/BLOCKSIZE1D, 1, 1);

	/* build the space of computation for random numbers and lattice simulation */
	s->prng_block = dim3(BLOCKSIZE1D, 1, 1);
	s->prng_grid = dim3( ((s->N/4) + BLOCKSIZE1D - 1)/BLOCKSIZE1D, 1, 1);

	/* alocate main arrays */
	malloc_arrays(s);

	/* reset table of obersvables per realization */
	reset_realization_statistics(s, s->R);


#ifdef MEASURE
	//  (1) log(n),  (2) sqrt(n) or (3) n/2   where 'n' is the number of ptsteps
	//int mzone = (ptsteps - (int)log2f((float)ptsteps));
	//s->mzone = (s->pts - (int)sqrtf((float)s->pts));
	//s->mzone = (int)sqrtf((float)s->pts);;
	//s->mzone = (int) (log((double)s->L) * sqrt((double)s->pts));
	//s->mzone = (int) (  (log(s->pts) / sqrt(s->pts)) * (double)s->pts);
	//s->mzone = (int) (( 1.0 - (2.0 / log( sqrt(s->pts) )) ) * (double)s->pts);
	//printf("%i zone \n", (int) (( 1.0 - (2.0 / log( sqrt(s->pts) )) ) * (double)s->pts));
	//printf("%i zone \n", (int) ( (1.0 / log( sqrt(s->pts) ) ) * (double)s->pts));
	//s->mzone = (int) (( (2.0 / log( sqrt(s->pts) )) ) * (double)s->pts);
	//s->mzone = 0;
	//s->mzone = 1000;
	//int mzone = s->pts/2;

	//printf("ignore the first %i steps of %i\n", s->mzone, s->pts);
#endif

	/* print parameters */
	//printparams(s);
    //printarray<float>(s->T, s->R, "T");
    //printarray<float>(s->E, s->R, "E");
	//printf("}:ok\n\n"); fflush(stdout);
}

/* malloc arrays */
void malloc_arrays( setup_t *s ){
	/* allocate the main arrays */
	s->hlat 	= (int **)malloc(sizeof(int *) * s->R);
	s->dlat 	= (int **)malloc(sizeof(int *) * s->R);

	/* T is a sorted temp array */
	s->T = (float*)malloc(sizeof(float)*s->R);
	/* ex is a per temperature counter array */
	s->ex = (float*)malloc(sizeof(float)*s->R);
	/* avex is a per temperature counter array */
	s->avex = (float*)malloc(sizeof(float)*s->R);
	/* index arrays */ 
	s->rts = (int*)malloc(sizeof(int)*s->R);
	s->trs = (int*)malloc(sizeof(int)*s->R);

	/* host values for each replica */
	s->E = (float*)malloc(sizeof(float)*s->R);
	s->exE = (float*)malloc(sizeof(float) * s->R);
	s->M = (int*)malloc(sizeof(int)*s->R);
	s->F1 = (float3*)malloc(sizeof(float3)*s->R);
	s->F2 = (float3*)malloc(sizeof(float3)*s->R);
	/* CUDA streams */
	s->rstream = (cudaStream_t*)malloc(sizeof(cudaStream_t) * s->R);
	/* PRNG states volume, one state per thread */
	s->pcga = (uint64_t **)malloc(sizeof(uint64_t *) * s->R);
	s->pcgb = (uint64_t **)malloc(sizeof(uint64_t *) * s->R);
	/* observables table */
	s->obstable = (obset_t*)malloc(sizeof(obset_t)*s->R);
	// memory for H array
	s->hH = (int*)malloc(sizeof(int) * s->N);
    
    /* global index of the first replica in each GPU */

	/* a copy of the magnetic field 'dH' on each GPU */
	s->dH = (int **)malloc(sizeof(int*) * s->ngpus);
	/* device values for GPUs */
	s->dE = (float**)malloc(sizeof(float*) * s->ngpus);
	s->dM = (int**)malloc(sizeof(int*) * s->ngpus);
	s->dF1 = (float3**)malloc(sizeof(float3*) * s->ngpus);
	s->dF2 = (float3**)malloc(sizeof(float3*) * s->ngpus);

	/* multi-GPU setup */
	#pragma omp parallel 
	{
		int tid, nt, r, k;
		/* set threads */
		threadset(s, &tid, &nt, &r);
		/* malloc the data for 'r' replicas on each GPU */
		for(int j = 0; j < r; ++j){
			k = tid * r + j;
			checkCudaErrors(cudaMalloc(&(s->dlat[k]), sizeof(int) * s->N));
			checkCudaErrors(cudaMalloc(&(s->pcga[k]), (s->N/4) * sizeof(uint64_t)));
			checkCudaErrors(cudaMalloc(&(s->pcgb[k]), (s->N/4) * sizeof(uint64_t)));
			checkCudaErrors(cudaStreamCreateWithFlags(&(s->rstream[k]), cudaStreamNonBlocking));
			kernel_gpupcg_setup<<<s->prng_grid, s->prng_block, 0, s->rstream[k] >>>(s->pcga[k], s->pcgb[k], s->N/4, s->seed, (unsigned long long)(SEQOFFSET*tid + k));
			//cudaDeviceSynchronize();
			cudaCheckErrors("kernel: prng reset");
		}	
		/* malloc device magnetic field -- multi-GPU */
		checkCudaErrors(cudaMalloc(&(s->dH[tid]), sizeof(int)*s->N));
		/* malloc device energy reductions -- multi-GPU*/ 
		checkCudaErrors(cudaMalloc(&(s->dE[tid]), sizeof(float)*r));
		checkCudaErrors(cudaMalloc(&(s->dM[tid]), sizeof(int)*r));
		checkCudaErrors(cudaMalloc(&(s->dF1[tid]), sizeof(float3)*r));
		checkCudaErrors(cudaMalloc(&(s->dF2[tid]), sizeof(float3)*r));
		/* P2P memory access is not working properly, for the moment just use standard device-host-device transfers */
		/* enable peer to peer memory access between GPUs */
		//if(tid != 0){
			//int access;
			//printf("\tGPU%i PeerAccess to GPU%i.....", s->gpus[tid].i, s->gpus[0].i); fflush(stdout);
			//checkCudaErrors(cudaDeviceCanAccessPeer(&access, s->gpus[tid].i, s->gpus[0].i));
			//printf("%i\n", access); fflush(stdout);
			//checkCudaErrors(cudaDeviceEnablePeerAccess( s->gpus[0].i, 0 )); 
		//}
		//else{
			//checkCudaErrors(cudaDeviceEnablePeerAccess( s->gpus[1].i, 0 )); 
		//}
	}
	/* host memory setup for each replica */
	for(int i = 0; i < s->R; i++){
		/* replica allocation */
		s->hlat[i]= (int*)malloc(sizeof(int) * s->N);

		/* array of temperatures increasing order */
		s->T[i] = s->TR - (s->R-1 - i)*s->dT;

		/* exchange counters initialization */
		s->ex[i] = 0;
		/* initialize index arrays */ 
		s->rts[i] = s->trs[i] = i;
	}
    int count  = 0;
    /* flatten the temperatures */
    for(int i=0; i<s->ngpus; ++i){
        for(int j=0; j<s->gpur[i]; ++j){
            s->T[count++] = s->aT[i][j];
        }
    }
    printarray<float>(s->T, s->R, "T");
    printf("\n");
}

/* pick the idlest 'n' gpus */
void pickgpus( setup_t *s ){ 
	/* structs for handling GPU queries error codes */
	nvmlReturn_t r;
	/* some function variables */
	unsigned int devcount, i, u;
	/* struct with GPU information */
	gpu_t *gpus;
	char version[80];
	/* init nvml library for GPU queries */
	r = nvmlInit(); 
	nvml_check(r, "nvmlInit");

	/* nvml: get driver version */
	r = nvmlSystemGetDriverVersion(version, 80); 
	nvml_check(r, "nvmlSystemGetDriverVersion");
	printf("\n\tDriver version: %s \n", version);

	/* get number of devices */
	r = nvmlDeviceGetCount(&devcount); 
	nvml_check(r, "nvmlDeviceGetCount");
	printf("\tMAXGPUS = %d\n", devcount);

	/* malloc one gpu_t struct for each device */
	gpus = (gpu_t*)malloc(sizeof(gpu_t)*devcount);
	/* return error if n > devcount */
	if( s->ngpus > devcount){
		fprintf(stderr, "pt error: [g = %i] > [MAXGPUS = %i]. (try g <= MAXGPUS)\n", s->ngpus, devcount);
		exit(1);
	}

	/* get the information of each GPU */
	printf("\tListing devices:\n");
	for(i = 0; i < devcount; i++){
		nvmlDevice_t dev;
		char name[64];
		//nvmlComputeMode_t compute_mode;
		nvmlUtilization_t util;
		r = nvmlDeviceGetHandleByIndex(i, &dev); 
		nvml_check(r, "nvmlDeviceGetHandleByIndex");
		r = nvmlDeviceGetName(dev, name, sizeof(name)/sizeof(name[0])); 
		nvml_check(r, "nvmlDeviceGetName");
		printf("\t\tGPU%d. %s", i, name);
		r = nvmlDeviceGetUtilizationRates(dev, &util); 
		u = nvml_check(r, "nvmlDeviceGetUtilizationRates");
		if(u){
			printf("  -> util = %i%%\n", util.gpu);
			gpus[i].i = i;
			gpus[i].u = util.gpu;
			gpus[i].m = util.memory;
		}
		else{
			gpus[i].i = i;
		}
	}
	if(u){
		//printf("not sorted\n");
		//for(i = 0; i < devcount; i++)
		//	printf("gpu[%i] = (i,u,m) ---> (%i, %i, %i)\n", i, gpus[i].i, gpus[i].u, gpus[i].m);
		//printf("sorted\n");
		qsort(gpus, devcount, sizeof(gpu), compgpu);
		//for(i = 0; i < devcount; i++)
		//	printf("gpu[%i] = (i,u,m) ---> (%i, %i, %i)\n", i, gpus[i].i, gpus[i].u, gpus[i].m);
	}
	/* malloc info for 'n' GPUs */
	s->gpus = (gpu_t*)malloc(sizeof(gpu_t)*s->ngpus);
	printf("\tchosen GPU(s) = {");
	for(i = 0; i < s->ngpus; i++){
		s->gpus[i] = gpus[i];
		printf(" GPU%i", s->gpus[i].i);
	}
	printf(" }\n");
	/* shutdown the nvml library */
	r = nvmlShutdown();
	nvml_check(r, "nvmlShutdown");
	/* free the auxiliary gpu_t array */
	free(gpus);
}

/* print parameters */
void printparams(setup_t *s){
	printf("\tparameters:{\n");
	printf("\t\tL:                            %i\n", s->L);
	printf("\t\tvolume:                       %i\n", s->N);
	printf("\t\t[TR,dT]:                      [%f, %f]\n", s->TR, s->dT);
	printf("\t\t[atrials, ains, apts, ams]:   [%i, %i, %i, %i]\n", s->atrials, s->ains, s->apts, s->ams);
	printf("\t\tmag_field h:                  %f\n", s->h);
	printf("\t\treplicas:                     %i\n", s->R);
	printf("\t\tptsteps:                      %i\n", s->pts);
	printf("\t\tmzone:                        %i\n", s->mzone);
	printf("\t\tdrop_steps:                   %i\n", s->ds);
	printf("\t\tmcsteps:                      %i\n", s->ms);
	printf("\t\tmeasure:                      %i\n", s->fs);
	printf("\t\tperiod:                       %i\n", s->period);
	printf("\t\tnblocks:                      %i\n", s->blocks);
	printf("\t\trealizations:                 %i\n", s->realizations);
	printf("\t\tseed:                         %i\n", s->seed);
	printf("\t\tmicrosteps:                   %i\n", s->cs);
	printf("\t\tNGPUS:                        %i\n\t}\n", s->ngpus);

	/* print space of computation */
	printf("\tsoc{\n\t\tmcgrid is %i x %i x %i  mcblock %i x %i x %i\n\t\tlgrid is %i x %i x %i  lblock %i x %i x %i \n\t}\n", 	
			s->mcgrid.x, s->mcgrid.y, s->mcgrid.z, s->mcblock.x, s->mcblock.y, s->mcblock.z,
			s->lgrid.x, s->lgrid.y, s->lgrid.z, s->lblock.x, s->lblock.y, s->lblock.z);
}

/* get parameters */
void getparams(setup_t *s, int argc, char **argv){
	/* if the number or arguments is not correct, stop the program */
	if(argc != 26){
		printf("run as:\n./bin/trueke -l <L> <R> -t <T> <dT> -a <tri> <ins> <pts> <ms> -h <h> -s <pts> <mz> <eq> <ms> <meas> <per> -br <b> <r> -g <x>\n");
		exit(1);
	}
	else{
		for(int i=0; i<argc; i++){
			/* lattice size and number of replicas */
			if(strcmp(argv[i],"-l") == 0){
				s->L = atoi(argv[i+1]);	
				s->R = atoi(argv[i+2]);
			}
			/* get TR and dT */
			else if(strcmp(argv[i],"-t") == 0){
				s->TR = atof(argv[i+1]);
				s->dT = atof(argv[i+2]);
			}
			/* the magnetic field constant */
			else if(strcmp(argv[i],"-h") == 0){
				s->h = atof(argv[i+1]);
			}
			/* ptsteps, drop steps, mc steps, final steps */
			else if(strcmp(argv[i],"-s") == 0){
				s->pts = atof(argv[i+1]);
                s->mzone = atoi(argv[i+2]);
				s->ds = atof(argv[i+3]);
				s->ms = atof(argv[i+4]);
				s->fs = atof(argv[i+5]);
				s->period = atof(argv[i+6]);
			}
			/* number of measure blocks and realizations */
			else if(strcmp(argv[i],"-br") == 0){
				s->blocks = atof(argv[i+1]);
				s->realizations = atof(argv[i+2]);
			}	
			/* adaptative dt parameters */
			else if(strcmp(argv[i], "-a") == 0){
				s->atrials = atoi(argv[i+1]);
                s->ains = atoi(argv[i+2]);
				s->apts = atoi(argv[i+3]);
				s->ams = atoi(argv[i+4]);
			}
			/* number of gpus */
			else if(strcmp(argv[i],"-g") == 0){
				s->ngpus = atoi(argv[i+1]);
			}
		}
	}
	if( (s->L % 32) != 0 )
		fprintf(stderr, "lattice dimensional size must be multiples of 32");
}

/* generate new seed */
void newseed(setup_t* s){
	//s->nseed = time(NULL);
}
#endif
