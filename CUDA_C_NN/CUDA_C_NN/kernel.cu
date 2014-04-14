
 
//#include "stdafx.h"
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h> 
#include <time.h>
#include <math.h>
#include <fcntl.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define NUMPAT 50
#define NUMIN  10
#define NUMHID 10
#define NUMOUT 1
#define SMALLWT 0.5

#define rando() ((double)rand()/(RAND_MAX+1))



void dmulWithCuda(double c[NUMIN + 1], double a[NUMIN + 1], double b[NUMIN + 1]);



__global__ void dmulKernel(double *c, double *a, double *b)
{
    int id = (blockIdx.x * blockDim.x) + threadIdx.x;
    if(id < (NUMIN + 1) )
	c[id] = a[id] * b[id];
	
}



int main() {
    int    i, j, k, p, np, op, ranpat[NUMPAT+1], epoch;
    int    NumPattern = NUMPAT, NumInput = NUMIN, NumHidden = NUMHID, NumOutput = NUMOUT;
    
	double Input[NUMPAT+1][NUMIN+1] = { 0,0,0,0,0,0,0,0,0,0,0,
										0,1,1,0,1,0,1,0,1,0,0,
										0,0,0,1,1,0,0,1,0,0,0,
										0,0,0,0,1,1,0,0,0,0,0,
										0,1,0,0,1,1,0,0,1,1,1,
										0,1,0,0,0,1,0,1,1,0,1,
										0,0,0,1,1,0,1,0,0,0,1,
										0,0,0,1,1,0,0,0,0,0,1,
										0,1,1,1,0,1,0,0,1,1,0,
										0,1,0,0,0,0,0,1,0,0,0,
										0,0,0,0,0,1,1,1,1,0,0,
										0,1,0,1,1,1,1,0,1,0,0,
										0,1,1,0,0,1,1,0,1,0,1,
										0,1,0,1,0,1,0,1,0,1,1,
										0,1,1,1,1,1,0,0,0,1,0,
										0,1,1,0,0,1,0,1,1,0,0,
										0,1,1,0,0,1,1,0,1,0,0,
										0,0,1,0,0,0,0,1,1,1,1,
										0,0,1,0,0,0,1,1,0,0,1,
										0,1,0,0,0,0,1,0,1,0,1,
										0,0,1,1,0,1,0,1,0,1,0,
										0,0,1,0,1,1,1,1,0,0,0,
										0,1,0,1,0,1,1,0,0,0,0,
										0,1,0,1,0,0,0,0,0,1,0,
										0,1,0,0,1,0,0,1,1,1,1,
										0,0,1,0,0,1,1,1,0,1,0,
										0,1,0,1,0,1,1,1,1,1,1,
										0,1,0,0,1,0,1,0,1,1,1,
										0,0,0,1,0,0,0,1,0,0,1,
										0,0,1,0,1,0,0,0,1,1,0,
										0,0,1,0,0,0,1,0,1,0,0,
										0,0,1,1,0,1,1,1,0,1,1,
										0,0,1,0,1,1,0,0,1,0,1,
										0,0,0,1,1,0,1,1,0,0,1,
										0,1,0,0,1,0,1,0,0,1,0,
										0,0,1,1,1,0,0,0,0,0,0,
										0,1,1,0,1,1,1,1,1,1,1,
										0,0,0,1,0,1,1,0,0,0,1,
										0,0,0,0,0,0,1,0,1,1,1,
										0,0,1,1,1,1,1,0,0,1,1,
										0,1,0,1,1,1,1,0,1,1,0,
										0,1,1,0,0,1,1,0,1,0,1,
										0,0,0,0,0,1,0,1,0,0,1,
										0,1,0,0,1,1,1,1,0,1,0,
										0,0,1,0,1,1,0,1,1,0,0,
										0,0,0,0,1,0,0,0,1,0,1,
										0,1,1,1,0,1,1,0,1,1,1,
										0,0,1,1,0,0,1,0,0,1,0,
										0,0,1,0,0,0,1,1,1,0,0,
										0,1,1,1,0,1,1,0,1,0,0,
										0,1,0,0,1,1,1,1,0,0,0,

                                                                 };     //0th row and 0th column are to be ignored
	                                                                    //Increase window width of cmd to see full output


    double Target[NUMPAT+1][NUMOUT+1] = { 0,0,
										  0,1,
										  0,1,
										  0,0,
										  0,0,
										  0,1,
										  0,0,
										  0,1,
										  0,0,
										  0,0,
										  0,0,
										  0,0,
										  0,0,
										  0,0,
										  0,0,
										  0,1,
										  0,1,
										  0,1,
										  0,0,
										  0,0,
										  0,1,
										  0,1,
										  0,0,
										  0,1,
										  0,0,
										  0,1,
										  0,0,
										  0,0,
										  0,1,
										  0,0,
										  0,1,
										  0,1,
										  0,1,
										  0,1,
										  0,0,
										  0,1,
										  0,1,
										  0,0,
										  0,0,
										  0,1,
										  0,1,
										  0,0,
										  0,1,
										  0,0,
										  0,1,
										  0,1,
										  0,0,
										  0,0,
										  0,0,
										  0,0,
										  0,1,
                                                   };      //0th row and 0th column are to be ignored    

    double SumH[NUMPAT+1][NUMHID+1], WeightIH[NUMIN+1][NUMHID+1], Hidden[NUMPAT+1][NUMHID+1];
    double SumO[NUMPAT+1][NUMOUT+1], WeightHO[NUMHID+1][NUMOUT+1], Output[NUMPAT+1][NUMOUT+1];
    double DeltaO[NUMOUT+1], SumDOW[NUMHID+1], DeltaH[NUMHID+1];
    double DeltaWeightIH[NUMIN+1][NUMHID+1], DeltaWeightHO[NUMHID+1][NUMOUT+1];
    double Error, eta = 0.5, alpha = 0.9, smallwt = 0.5;
  
    for( j = 1 ; j <= NumHidden ; j++ ) {    // initialize WeightIH and DeltaWeightIH 
        for( i = 0 ; i <= NumInput ; i++ ) { 
            DeltaWeightIH[i][j] = 0.0 ;
            WeightIH[i][j] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
	}

    for( k = 1 ; k <= NumOutput ; k ++ ) {    /* initialize WeightHO and DeltaWeightHO */
        for( j = 0 ; j <= NumHidden ; j++ ) {
            DeltaWeightHO[j][k] = 0.0 ;              
            WeightHO[j][k] = 2.0 * ( rando() - 0.5 ) * smallwt ;
        }
    }
     
    for( epoch = 0 ; epoch < 10000 ; epoch++) {    /* iterate weight updates */

		double host_c[NUMIN + 1], host_a[NUMIN + 1],host_b[NUMIN + 1];


        for( p = 1 ; p <= NumPattern ; p++ ) {    /* randomize order of individuals */
            ranpat[p] = p ;
        }
        for( p = 1 ; p <= NumPattern ; p++) {
            np = p + rando() * ( NumPattern + 1 - p ) ;
            op = ranpat[p] ; ranpat[p] = ranpat[np] ; ranpat[np] = op ;
        }
        Error = 0.0 ;
        for( np = 1 ; np <= NumPattern ; np++ ) {    /* repeat for all the training patterns */
            p = ranpat[np];
            
			
			
			for( j = 1 ; j <= NumHidden ; j++ ) {    /* compute hidden unit activations */
                SumH[p][j] = WeightIH[0][j] ;
                
				
				for( int ctr = 1 ; ctr <= NumInput ; ctr++ ) {

					host_a[ctr] = Input[p][ctr];
					host_b[ctr] = WeightIH[ctr][j];
				}
				
				dmulWithCuda(host_c, host_a, host_b);
				
				for( i = 1 ; i <= NumInput ; i++ ) {
                    
					//host_c[i] = host_a[i] * host_b[i];
					//val[i] = Input[p][i] * WeightIH[i][j];
					SumH[p][j] +=  host_c[i]; //<<<Parallelize this>>>
                
				}
				
                
				
				
				
				Hidden[p][j] = 1.0/(1.0 + exp(-SumH[p][j])) ;
            }

			
            for( k = 1 ; k <= NumOutput ; k++ ) {    /* compute output unit activations and errors */
                SumO[p][k] = WeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    SumO[p][k] += Hidden[p][j] * WeightHO[j][k] ;  //<<<Parallelize this>>>
                }
                Output[p][k] = 1.0/(1.0 + exp(-SumO[p][k])) ;   /* Sigmoidal Outputs */
/*              Output[p][k] = SumO[p][k];      Linear Outputs */
                Error += 0.5 * (Target[p][k] - Output[p][k]) * (Target[p][k] - Output[p][k]) ;   /* SSE */
/*              Error -= ( Target[p][k] * log( Output[p][k] ) + ( 1.0 - Target[p][k] ) * log( 1.0 - Output[p][k] ) ) ;    Cross-Entropy Error */
                DeltaO[k] = (Target[p][k] - Output[p][k]) * Output[p][k] * (1.0 - Output[p][k]) ;   /* Sigmoidal Outputs, SSE */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Sigmoidal Outputs, Cross-Entropy Error */
/*              DeltaO[k] = Target[p][k] - Output[p][k];     Linear Outputs, SSE */
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {    /* 'back-propagate' errors to hidden layer */
                SumDOW[j] = 0.0 ;
                for( k = 1 ; k <= NumOutput ; k++ ) {
                    SumDOW[j] += WeightHO[j][k] * DeltaO[k] ;  //<<<Parallelize this>>>
                }
                DeltaH[j] = SumDOW[j] * Hidden[p][j] * (1.0 - Hidden[p][j]) ;
            }
            for( j = 1 ; j <= NumHidden ; j++ ) {     /* update weights WeightIH */
                DeltaWeightIH[0][j] = eta * DeltaH[j] + alpha * DeltaWeightIH[0][j] ;
                WeightIH[0][j] += DeltaWeightIH[0][j] ;
                for( i = 1 ; i <= NumInput ; i++ ) { 
                    DeltaWeightIH[i][j] = eta * Input[p][i] * DeltaH[j] + alpha * DeltaWeightIH[i][j]; //<<<Parallelize this>>>
                    WeightIH[i][j] += DeltaWeightIH[i][j] ; //<<<Parallelize this>>>
                }
            }
            for( k = 1 ; k <= NumOutput ; k ++ ) {    /* update weights WeightHO */
                DeltaWeightHO[0][k] = eta * DeltaO[k] + alpha * DeltaWeightHO[0][k] ;
                WeightHO[0][k] += DeltaWeightHO[0][k] ;
                for( j = 1 ; j <= NumHidden ; j++ ) {
                    DeltaWeightHO[j][k] = eta * Hidden[p][j] * DeltaO[k] + alpha * DeltaWeightHO[j][k] ; //<<<Parallelize this>>>
                    WeightHO[j][k] += DeltaWeightHO[j][k] ; //<<<Parallelize this>>>
                }
            }
        }
        if( epoch%100 == 0 ) fprintf(stdout, "\nEpoch %-5d :   Error = %f", epoch, Error) ;
        if( Error < 0.0004 ) break ;  /* stop learning when 'near enough' */
		
    }
    
    fprintf(stdout, "\n\nNETWORK DATA - EPOCH %d\n\nPat\t", epoch) ;   /* print network outputs */
    
	for( i = 1 ; i <= NumInput ; i++ ) {
        fprintf(stdout, "I%-4d\t", i) ;
    }
    for( k = 1 ; k <= NumOutput ; k++ ) {
        fprintf(stdout, "T%-4d\tO%-4d\t", k, k) ;
    }
    for( p = 1 ; p <= NumPattern ; p++ ) {        
    fprintf(stdout, "\n%d\t", p) ;
        for( i = 1 ; i <= NumInput ; i++ ) {
            fprintf(stdout, "%.1f\t", Input[p][i]) ;
        }
        for( k = 1 ; k <= NumOutput ; k++ ) {
            fprintf(stdout, "| %.1f\t%f\t", Target[p][k], Output[p][k]) ;
        }
    }
  fprintf(stdout, "\n\nGoodbye! \n\n") ;

  	

  return 1 ;
}

/*******************************************************************************/



// Helper function for using CUDA to multiply double vectors in parallel.
void dmulWithCuda(double c[NUMIN + 1], double a[NUMIN + 1], double b[NUMIN + 1])
{

	double *dev_a = 0;
    double *dev_b = 0;
    double *dev_c = 0;

	int size = NUMIN + 1;

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaMalloc((void**)&dev_c, size * sizeof(double));
    cudaMalloc((void**)&dev_a, size * sizeof(double));
    cudaMalloc((void**)&dev_b, size * sizeof(double));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_a, a, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, size * sizeof(double), cudaMemcpyHostToDevice);

    // Launch a kernel on the GPU with one thread for each element.
    dmulKernel<<<1,(NUMIN + 1)>>>(dev_c, dev_a, dev_b);
    
    // cudaDeviceSynchronize waits for the kernel to finish.
    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(c, dev_c, size * sizeof(double), cudaMemcpyDeviceToHost);

	// Free allocated memory on the GPU
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

}