#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>

#define MAX_ERR 1e-6

//    n : nombre de lignes de la matrice,
//    p : nombre de colonnes de la matrice si n différent de p,
//    M : pointeur de la matrice

// Sur CPU


// ------------------------------------------------ Macros --------------------------------------------------------------------

void MatrixInit(float *M, int n, int p); //Cette fonction initialise une matrice de taille n x p. Initialisez les valeurs de la matrice de façon aléatoire entre -1 et 1.
void MatrixPrint(float *M, int n, int p); // Cette fonction affiche une matrice de taille n x p
void MatrixAdd(float *M1, float *M2, float *Mout, int n, int p); //Cette fonction additionne deux matrices M1 et M2 de même taille n x p
void MatrixMult(float *M1, float *M2, float *Mout, int n); //Cette fonction multiplie 2 matrices M1 et M2 de taillle n x n. Ne pas dépasser N=1000
 
// Sur GPU

__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p); //Cette fonction additionne deux matrices M1 et M2 de même taille n x p. Vous pouvez considérer les dimensions des matrices comme les paramètres gridDim et blockDim : les lignes correspondent aux blocks, les colonnes correspondent aux threads

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n); //Cette fonction multiplie 2 matrices M1 et M2 de taillle n x n. Vous pouvez considérer les dimensions des matrices comme les paramètres gridDim et blockDim : les lignes correspondent aux blocks, les colonnes correspondent aux threads



//---------------------------------------------- Fonctions ------------------------------------------------------------------
void MatrixPrint(float *M, int n, int p){
    
    int i,j;
    
    printf("Two Dimensional array elements: \n");
       for(i=0; i<n; i++) {
          for(j=0;j<p;j++) {
              if(M[i*p +j]>0) printf(" ");
             printf("%1.5f ", M[i*p +j]);
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


void MatrixSub(float *M1, float *M2, float *Mout, int n, int p){
    
    int i,j;
    
    for (i=0; i<n; i++){
        for(j=0; j<p; j++){
            Mout[i*p + j]=M1[i*p + j]-M2[i*p + j];
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


__global__ void cudaMatrixAdd(float *M1, float *M2, float *Mout, int n, int p) {
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Handling arbitrary vector size
    if (i < n && j < p) {
        Mout[i*p + j] = M1[i*p + j] + M2[i*p + j];
    }
}

__global__ void cudaMatrixMult(float *M1, float *M2, float *Mout, int n) {
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;  
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < n && col < n) {
      for (int i = 0; i < n; ++i) {
          Mout[row * n + col] += M1[row * n + i] * M2[i * n + col];
      }
    }
}
    
    
  

//-------------------------------------------- main() --------------------------------------------------------------------- 

int main(int argc, char *argv[]){
    
    
    if (argc < 2) { // Si pas assez d'arguments en entrée
        printf("Usage: ./%s n p \n", argv[0]);
        exit(EXIT_FAILURE);
    }
    
    srand((unsigned int)time(NULL)); // Initialisation seed pour fonction random dans l'initilisation dans MatrixInit
    
    int n = atoi(argv[1]);           // Nombre de lignes 
    int p = atoi(argv[2]);           // Nombre de colonnes
    
    
    //---------------------------- Déclaration des matrices ------------------------------------------------- 
    
    //Sur CPU
    float* M1= (float*)malloc(sizeof(float) * n * p); 
    float* M2= (float*)malloc(sizeof(float) * n * p);
    float* MoutAdd= (float*)malloc(sizeof(float) * n * p);
    float* MoutAddGpu= (float*)malloc(sizeof(float) * n * p);
    float* MDiff= (float*)malloc(sizeof(float) * n * p);
    float* MoutMult= (float*)malloc(sizeof(float) * n * n);
    
    //Sur GPU
    float* d_M1; 
    float* d_M2;
    float* d_MoutAdd;
    float* d_MoutMult;
    
    cudaMalloc((void**)&d_M1, sizeof(float)*n*p);
    cudaMalloc((void**)&d_M2, sizeof(float)*n*p);
    cudaMalloc((void**)&d_MoutAdd, sizeof(float)*n*p);
    cudaMalloc((void**)&d_MoutMult, sizeof(float)*n*n);
    
    
    // ----------------------- Implémentation sur CPU -------------------------
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
    MatrixMult(M1, M2, MoutMult, n, n);
    MatrixPrint(MoutMult,n,p);
    
   
    
    // -----------------------Implémentation sur GPU------------------------------
    // Main function
    
    cudaMemcpy(d_M1, M1, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    cudaMemcpy(d_M2, M2, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    
    
    int block_size = 3;
    int grid_size = 7;
    cudaMatrixAdd<<<grid_size,block_size>>>(d_M1, d_M2, d_MoutAdd, n,p);
    cudaMatrixMult<<<grid_size,block_size>>>(d_M1, d_M2, d_MoutMult, n);
    
    cudaMemcpy(MoutAddGpu, d_MoutAdd, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
    cudaMemcpy(MoutMult, d_MoutMult, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    
    printf("Addition de M1 et M2 sur GPU\n");
    MatrixAdd(M1, M2, MoutAddGpu, n, p);
    MatrixPrint(MoutAddGpu,n,p);
    
    printf("Multiplication de M1 et M2 sur GPU\n");
    MatrixMult(M1, M2, MoutMult, n, n);
    MatrixPrint(MoutMult,n,p);
     
    //--diff
    
    MatrixSub(MoutAdd,MoutAddGpu,MDiff,n,n);
    MatrixPrint(MDiff,n,n);
    
    // ------------------------- Free ----------------------------------------
    free(M1);
    free(M2);
    free(MoutAdd);
    free(MoutMult);
    
    cudaFree(d_M1);
    cudaFree(d_M2);
    cudaFree(d_MoutAdd);
    cudaFree(d_MoutMult);
    
    cudaDeviceSynchronize();

    return 0;
    
}
