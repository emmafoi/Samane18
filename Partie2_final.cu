/******************************************* PARTIE 2 ************************************************/

/* L'objectif de cette deuxième partie est d'implémenter un petit réseau convolutionnel qui va 
*  convoluer une image d'entrée raw_data de taille 32*32 par un une série de 6 kernels de taille 5*5
*  stockés dans la matrice C1_kernels. Nous obtenons en sortie de cette couche 6 feature maps 
*  de taille 28*28 (la taille 28 résultant de l'opération (32 - 5 + 1)), qui nous donnent la matrice 
*  C1_data. La deuxième couche est une étape de sous-échantillonage par 2 qui va, sur chaque feature map,
*  moyenner chaque carré de pixels de taille de 2*2. Cela nous donne 6 matrices de taille 14*14, la 
*  taille des feature maps étant divisée par deux, et nous appliquons à chaque pixels de ces matrices
*  la fonction d'activation tanh (qui ramène les valeurs entre -1 et 1) et nous stockons les 6 matrices
*  14*14 dans la matrice S1_data.
*
*  Nous allons effectuer plusieurs tests pour montrer que notre code marche bien, en montrant les 
*  résultats d'abord sur des matrices simples de petite taille pour raw_data et C1_data, dont nous 
*  connaissons le résultat de la convolution et du sous-échantillonnage. Nous montrerons en dernier le 
*  fonctionnement du code avec les matrices raw_data et C1_kernel initialisées avec des nombres 
*  aléatoires entre 0 et 1.
*  
*  Le code comporte d'abord les fonctions que nous utilisons pour les différentes initialisations des 
*  matrices, puis les codes des fonctions de convolution, de sous-échantillonnage et de tanh. Nous avons
*  ensuite le main, dont on expliquera la structure plus bas dans le code, juste avant son début.
*/

// ----------------------------------------- Includes ------------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdlib>
#define BLOCK_SIZE 32


// ------------------------------- Fonctions d'initialisation de matrices ----------------------------

/* Ces premières fonctions nous aident à intialiser les matrices raw_data, C1_data, C1_kernel, et 
*  S1_data. Chaque fonction prend en argument 
*
*/

/*
 * Fonction: MatrixInitInt
 * ----------------------------
 *   Initialise chaque valeur M[i] d'un tableau M par la valeur i 
 *   M  : tableau à initialiser
 *   size : nombre total d'éléments dans M
 */
void MatrixInitInt(float *M,int size)
{
    for(int i=0;i<size;i++){
        M[i]=i;
    }
}

/*
 * Fonction: MatrixInitOne
 * ----------------------------
 *   Initialise chaque valeur M[i] d'un tableau M par la valeur 1
 *   M  : tableau à initialiser
 *   size : nombre total d'éléments dans M
 */

void MatrixInitOne(float *M,int size)
{
    for(int i=0;i<size;i++){
        M[i]=1;
    }
}

/*
 * Fonction: MatrixInitZero
 * ----------------------------
 *   Initialise chaque valeur M[i] d'un tableau M par la valeur 0 
 *   M  : tableau à initialiser
 *   size : nombre total d'éléments dans M
 */

void MatrixInitZero(float *M,int size)
{
    for(int i=0;i<size;i++){
        M[i]=0;
    }
}

void MatrixInitRand(float *M, int size){
    for (int i = 0; i<size; i++){
        M[i] = (float)(rand()%1000)/1000 ; 
        //flottant entre 0 et 1 de précision 10⁻3
    }
}


void MatrixPrint(float *C,const int nx,const int ny,const int nz)
{
    float *ic=C;
    printf("\n Matrix: (%d*%d*%d) \n",nx,ny,nz);
    for(int k=0;k<nz;k++){
        for(int i=0;i<ny;i++){
            for(int j=0;j<nx;j++){
                if(ic[k*(nx*ny)+nx*i +j]<0){
                    printf("%3.1f ",ic[k*(nx*ny)+nx*i +j]);                     
                }else{
                    printf(" %3.1f ",ic[k*(nx*ny)+nx*i +j]);
                }
            }
            printf("\n");

        }
        printf("\n");
    }
}

// Fonction qui sert à afficher l'indexage globale d'un thread sur une grille, pour mieux comprendre l'indexage 
// A servi simplement de helper function
__global__ void printthreadindex(float *A,const int nx,const int ny)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    //int iz=threadIdx.z+blockIdx.z*blockDim.z;
    
    
    unsigned int idx=ix+iy*nx; //+nx*ny*iz;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index %2d  ival %2d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);

}


// --------------------------------- Convolution, sous-échantillonnage, et fonction d'activation ------------------------------------

// Fonction exécutée par le device (GPU) et appelée par le device (GPU).
// Elle doit être appellée dans un kernel et ne nécessite d'appel <<<B,T>>> comme les fonctions __global__
// On l'appelle à la fin de la fonction cudaMoyen2 
__device__ float activation_tanh(float M){
    return tanhf(M);
}


// Fonction qui réaliser la convolution d'une matrice avec 1 SEUL kernel, donne en sortie 1 seul feature map
__global__ void gpuMatrix2DConv(float* gpuMat1, float* kernel, float* gpuMat3, int m1Rows, int m1Cols, int mRowsCols, int m3Rows, int m3Cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if (row < m3Rows && col < m3Cols) {
        for (int maskRow = 0; maskRow < mRowsCols; maskRow++) {
            for (int maskCol = 0; maskCol < mRowsCols; maskCol++) {
                sum += gpuMat1[(row + maskRow) * m1Cols + (col + maskCol)] * kernel[maskRow * mRowsCols + maskCol];
            }
        }
        gpuMat3[row * m3Cols + col] = sum;
    }
}

// On part de la fonction de convolution 2D pour construire la convolution 3D : à chaque thread, on calcule la valeur du pixel (i,j) de CHAQUE feature map. Un thread réalise donc nb_features_maps (ici 6) convolutions avec un carré de l'image
__global__ void gpuMatrix3DConv(float* gpuMat1, float* kernel, float* gpuMat3, int m1Rows, int m1Cols, int kernel_size, int nb_kernels, int m3Rows, int m3Cols){
    
    //Identifiants globaux ligne et colonne de la matrice gpuMat1
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m3Rows && col < m3Cols) {
        
        // On itère l'opération de convolution d'un carré kernel_size*kernel_size de la matrice d'entrée sur chacun des nb_kernels kernels 
        for(int num_kernel = 0; num_kernel<nb_kernels; num_kernel++){
            // L'offset ici correpond au numéro de kernel dans avec lequel on fait une opération
            int offset = num_kernel*kernel_size*kernel_size;
            //On initialise la somme qui donnera la valeur final du pixel (row,col) de la matrice C1_data
            float sum = 0.0;
            for (int maskRow = 0; maskRow < kernel_size; maskRow++) {
                for (int maskCol = 0; maskCol < kernel_size; maskCol++) {
                    sum += gpuMat1[(row + maskRow) * m1Cols + (col + maskCol) ] * kernel[maskRow * kernel_size + maskCol + offset];
                }
            }
            gpuMat3[num_kernel*(m3Rows*m3Cols) + row * m3Cols + col] = sum;
        
        }
    }
    
}

/* fonction moyenneur executée sur GPU 
L'argument de la fonction correspond à la dimension de la matrice d'entrée
Les nombres de blocks et threads sont ceux de la matrice d'arrivés car le nombre de calculs corespond au nombre d'éléments à l'arrivée
n : taille de la matrice d'entrée
*/
__global__ void cudaMoyen2(float *E, float *S, int n){
    // n = taille d'une ligne de E (et aussi d'une colonne)
    
    int n_out = n/2; // dimension de la matrice de sortie
    
    //1er élément du 1er dim3 = nombre matrices 2D de E
    int nb_mat = blockIdx.x;
    
    //nb_mat * taille d'une matrice de S (= taille du shift dans l'indice de S):
    int shift_S = nb_mat * n_out * n_out ;
    //nb_mat * taille d'une matrice de E (= taille du shift dans l'indice de E):
    int shift_E = nb_mat * n * n ;
    
    //2e élément du 1er dim3 = nombre de colonnes/2 de E = nombre de col de S
    int output_col = blockIdx.y; 
    
    //2e dim3 (contient 1 seul élément) = nombre de lignes/2 de E =  nombre de lignes de S
    int output_row = threadIdx.x;
    
    //on se déplace de 2 en 2 dans les matrices d'entrée
    int input_col = 2 * output_col;
    int input_row = 2 * output_row;
    
    //Calcul pour chaque élément de S la moyenne en fonction des éléments de E :
    S[shift_S + output_row * n_out + output_col] = (float)(( E[shift_E + input_row * n + input_col] + E[shift_E + (input_row+1) * n + input_col] + E[shift_E + input_row * n + (input_col+1)] + E[shift_E + (input_row+1) * n + (input_col+1)] )/4);
    
}



int main()
{
    //Matrix raw_data
    int raw_size=8;
    float rawBytes=raw_size*raw_size*sizeof(float);

    float *raw_data;
    raw_data=(float *)malloc(rawBytes);

    MatrixInitOne(raw_data,raw_size*raw_size);
    MatrixPrint(raw_data,raw_size,raw_size,1);
    
    float *d_raw_data;
    cudaMalloc((void **)&d_raw_data,rawBytes);

    cudaMemcpy(d_raw_data,raw_data,rawBytes,cudaMemcpyHostToDevice);
    
    
    //Matrix C1_Kernel
    int C1_kernel_size=3,nb_kernels=6;
    float C1_kernelBytes=C1_kernel_size*C1_kernel_size*nb_kernels*sizeof(float);
    
    float *C1_kernel;
    C1_kernel=(float *)malloc(C1_kernelBytes);
    
    MatrixInitInt(C1_kernel,C1_kernel_size*C1_kernel_size*nb_kernels);
    MatrixPrint(C1_kernel,C1_kernel_size,C1_kernel_size,nb_kernels);
    
    float *d_C1_kernel;
    cudaMalloc((void **)&d_C1_kernel,C1_kernelBytes);

    cudaMemcpy(d_C1_kernel,C1_kernel,C1_kernelBytes,cudaMemcpyHostToDevice);
    
    
    //Matrix C1_data output of convolution 1
    int C1_data_size=6,nb_of_maps=6;
    float C1_data_Bytes=C1_data_size*C1_data_size*nb_of_maps*sizeof(float);
    
    float *C1_data;
    C1_data=(float *)malloc(C1_data_Bytes);

    float *d_C1_data;
    cudaMalloc((void **)&d_C1_data,C1_data_Bytes);

    MatrixInitZero(C1_data,C1_data_size*C1_data_size*nb_of_maps);
    MatrixPrint(C1_data,C1_data_size,C1_data_size,nb_of_maps);
    
    cudaMemcpy(d_C1_data,C1_data,C1_data_Bytes,cudaMemcpyHostToDevice);
    
    //Matrix S1_data output of ss-ech 1
    int S1_data_size=3;
    float S1_data_Bytes=S1_data_size*S1_data_size*nb_of_maps*sizeof(float);
    
    float *S1_data;
    S1_data=(float *)malloc(S1_data_Bytes);

    float *d_S1_data;
    cudaMalloc((void **)&d_S1_data,S1_data_Bytes);

    MatrixInitZero(S1_data,S1_data_size*S1_data_size*nb_of_maps);
    MatrixPrint(S1_data,S1_data_size,S1_data_size,nb_of_maps);
    
    cudaMemcpy(d_S1_data,S1_data,S1_data_Bytes,cudaMemcpyHostToDevice);
    
    // Process
    //dim3 block(raw_size);//,3);
    //dim3 grid(raw_size);
    //printthreadindex <<<grid,block>>> (d_MatA,nx,ny);//,1);
    
    //printthreadindex <<<grid,block>>> (d_raw_data,raw_size,raw_size);//,1);

    
    // --------------- Convolution 1 -----------------------------
    int threadsPerBlock = 32;
    int gridCols = ceil(double(C1_data_size) / double(threadsPerBlock));
    int gridRows = ceil(double(C1_data_size) / double(threadsPerBlock));

    dim3 gridDim(gridCols, gridRows);
    dim3 blockDim(threadsPerBlock, threadsPerBlock);	// total 32x32=1024 threads
    //gpuMatrix2DConv << < gridDim, blockDim >> > (d_raw_data, d_C1_kernel, d_C1_data, raw_size, raw_size, C1_kernel_size, C1_data_size, C1_data_size);
    gpuMatrix3DConv << < gridDim, blockDim >> > (d_raw_data, d_C1_kernel, d_C1_data, raw_size, raw_size, C1_kernel_size,nb_kernels, C1_data_size, C1_data_size);
    
    // ------------------ Sous-échantillonage 1 ------------------------------
    dim3 my_blocks (nb_of_maps, S1_data_size, 1);
    cudaMoyen2<<<my_blocks,S1_data_size>>>(d_C1_data,d_S1_data, C1_data_size);
    
    
    
    // Retour au CPU
    cudaMemcpy(C1_data, d_C1_data, C1_data_Bytes, cudaMemcpyDeviceToHost); // C1_data
    cudaMemcpy(S1_data, d_S1_data, S1_data_Bytes, cudaMemcpyDeviceToHost); // C1_data
    
    // Affichage Conv et Sous-échantillonage 
    printf("\nConvolution\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size,nb_of_maps);

    printf("\nSous-échantillonage\n");
    MatrixPrint(S1_data,S1_data_size,S1_data_size,nb_of_maps);

    // Libération des ressources 
    cudaFree(d_raw_data);
    cudaFree(d_C1_kernel);
    cudaFree(d_C1_data);
    
    free(raw_data);
    free(C1_kernel);
    free(C1_data);
    
    // This call waits for all of the submitted GPU work to complete
    cudaDeviceSynchronize();
    
    return 0;

}