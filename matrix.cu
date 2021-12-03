#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

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
}


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

void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p){
    
    int i,j;
    
    for (i=0; i<n; i++){
        for(j=0; j<p; j++){
            Mout[i*p + j]=M1[i*p + j] +M2[i*p + j];
        }
    }
    
}
    

void MatrixMult(float *M1, float *M2, float *Mout, int n){
    
    
}


int main(int argc, char *argv[]){
    
    if (argc < 2) {
        printf("Usage: ./%s n p \n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    int n = atoi(argv[1]);
    int p = atoi(argv[2]);
    
    float* M1= (float*)malloc(sizeof(float) * n *p);
    float* M2= (float*)malloc(sizeof(float) * n *p);
    float* Mout= (float*)malloc(sizeof(float) * n *p);
    
    MatrixInitfloat(M1, n, p);
    MatrixPrint(M1,n,p);
    
    MatrixInitfloat(M2, n, p);
    MatrixPrint(M2,n,p);
    
    MatrixAdd(M1, M2, Mout, n, p);
    MatrixPrint(Mout,n,p);
    return 0;
    
}
