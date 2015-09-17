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

# trueke: A multi-GPU Monte Carlo simulator for the 3D random field Ising model

# (1) Hardware requirements:
- A CUDA capable GPU, we recommend Kepler+
- 4GB of RAM
- Multi-core X86_64 CPU




# (2) Software requirements:
- Linux OS
- GCC.
- Nvidia CUDA runtime library
- Nvidia nvcc compiler
- [optional] Nvidia nvlm (to query the device).
- OpenMP 4.0 Implementation






# (3) Check that the Makefile is set up according to your CUDA instalation. 
If not, re-write the bin, inc and lib paths to the correspondng ones.







# (4) compilation (edit Makefile if necessary)
$ make clean
$ make






# (5) how to run
./bin/trueke -l \<L\> <R> -t <T> \<dT\> -a <tri> <\ins\> <pts> <ms> -h <h> -s <pts> <mz> <eq> <ms> <meas> <per> -br <\b> <r> -g <x>






# (6) parameters
-l <L> <R>                             Lattice:        size <L>, <R> replicas.
-t <T> <dT>                            Temperature:    highest <T>, delta <dT>.
-a <tri> <ins> <pts> <ms>              Adaptive:       <tri> trials, <\ins> inserts/trial, <pts> exchange steps, <ms> sweeps per exchange.
-h <h>                                 External Field: magnetic field strength <h> (tipically, 0 < h < 3).
-s <pts> <mz> <eq> <ms> <meas> <per>   Simulation:     <pts> exchange steps, measure from <mz>, equilibrate <eq> sweeps, <ms> sweeps/exchange, <per> ex/<meas>
-br <\b> <r>                           Repetition:     <\b> blocks of <ms>, <r> disorder realizations.
-g <x>                                 Multi-GPU:      use <x> GPUs.






# (4) Example execution using two GPUs 
bin/trueke -l 64 11 -t 4.7 0.1 -a 58 2 2000 10 -h 1.0 -s 5000 3000 100 5 1 1 -br 1 2000 -g 2
