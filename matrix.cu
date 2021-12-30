#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

//    n : nombre de lignes de la matrice,
//    p : nombre de colonnes de la matrice si n différent de p,
//    M : pointeur de la matrice

// Sur CPU

void MatrixInit(float *M, int n, int p); //Cette fonction initialise une matrice de taille n x p. Initialisez les valeurs de la matrice de façon aléatoire entre -1 et 1.
void MatrixPrint(float *M, int n, int p); // Cette fonction affiche une matrice de taille n x p
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p); //Cette fonction additionne deux matrices M1 et M2 de même taille n x p
void MatrixMult(float *M1, float *M2, float *Mout, int n); //Cette fonction multiplie 2 matrices M1 et M2 de taillle n x n. Ne pas dépasser N=1000
 
// Sur GPU

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p); //Cette fonction additionne deux matrices M1 et M2 de même taille n x p. Vous pouvez considérer les dimensions des matrices comme les paramètres gridDim et blockDim : les lignes correspondent aux blocks, les colonnes correspondent aux threads

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n); //Cette fonction multiplie 2 matrices M1 et M2 de taillle n x n. Vous pouvez considérer les dimensions des matrices comme les paramètres gridDim et blockDim : les lignes correspondent aux blocks, les colonnes correspondent aux threads

void MatrixPrint(float *M, int n, int p){
    
    int i,j;
    
    printf("Two Dimensional array elements: \n");
       for(i=0; i<n; i++) {
          for(j=0;j<p;j++) {
             printf("%f ", M[i*p +j]);
             if(j==n-1){
                printf("\n");
             }
          }
       }
    printf("\n");
}


/* Initialisation en matrice identité
void MatrixInitfloat (float *M, int n, int p){
    int i,j;
    
    for (i=0; i<n; i++){
        for(j=0; j<p; j++){
            
            if( i == j ){
                M[i*p + j]=1;
            }
            else{
                M[i*p + j]=0;
            }   
        }
    }
}
*/

// Initialisation M1 = [[1,2]     et   M2 = [[5,6] 
//                      [3,4]]               [7,8]]

/*
int k=1;
void MatrixInitfloat (float *M, int n, int p){
    int i,j;

    
    for (i=0; i<n; i++){
        for(j=0; j<p; j++){
            M[i*p + j] = k;
            k+=1;
        }
    }
}
*/

void MatrixInitfloat (float *M, int n, int p){
    int i,j;
    float upper_bound = 2.0;  // we generate a random value in [0,2] and substract 1 to be in [-1,1] 
    
    for (i=0; i<n; i++){
        for(j=0; j<p; j++){
            
            M[i*p + j] = ((float)rand()/(float)(RAND_MAX)) * upper_bound -1.0;
        }
    }
}

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    int i,j;
    
    for (i=0; i<n; i++){
        for(j=0; j<p; j++){
            Mout[i*p + j]=M1[i*p + j] +M2[i*p + j];
        }
    }
    
}
    

void MatrixMult(float *M1, float *M2, float *Mout, int n, int p){
    
    int i,j;
    for(i=0;i<n;i++){
        for(j=0;j<p;j++){
                Mout[i*p + j]=0;
                for(int k=0;k<p;k++)
                    Mout[i*p + j]+= M1[i*p + k]*M2[k*p + j];
        }
    }
    
}


int main(int argc, char *argv[]){
    
    if (argc < 2) {
        printf("Usage: ./%s n p \n", argv[0]);
        exit(EXIT_FAILURE);
    }
    srand((unsigned int)time(NULL));
    
    int n = atoi(argv[1]);
    int p = atoi(argv[2]);
    
    float* M1= (float*)malloc(sizeof(float) * n * p);
    float* M2= (float*)malloc(sizeof(float) * n * p);
    float* MoutAdd= (float*)malloc(sizeof(float) * n * p);
    float* MoutMult= (float*)malloc(sizeof(float) * n * p);
    
    printf("Initialisation et affichage de M1 \n");
    MatrixInitfloat(M1, n, p);
    MatrixPrint(M1,n,p);
    
    printf("Initialisation et affichage de M2 \n");
    MatrixInitfloat(M2, n, p);
    MatrixPrint(M2,n,p);
    
    printf("Addition de M1 et M2 sur CPU\n");
    MatrixAdd(M1, M2, MoutAdd, n, p);
    MatrixPrint(MoutAdd,n,p);
    
    printf("Multiplication de M1 et M2 sur CPU\n");
    MatrixMult(M1, M2, MoutMult, n, p);
    MatrixPrint(MoutMult,n,p);
    
    free(M1);
    free(M2);
    free(MoutAdd);
    free(MoutMult);
    
    return 0;
    
}


