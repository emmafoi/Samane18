/******************************************* PARTIE 2 ************************************************/

/* L'objectif de cette deuxième partie est d'implémenter un petit réseau convolutionnel qui va 
*  convoluer une image d'entrée raw_data de taille 32*32 par un une série de 6 kernels de taille 5*5
*  stockés dans la matrice C1_kernels. Nous obtenons en sortie de cette couche 6 feature maps 
*  de taille 28*28 (la taille 28 résultant de l'opération (32 - 5 + 1)), auxquelles nous appliquons 
*  à chaque pixels de ces matrices la fonction d'activation tanh (qui ramène les valeurs entre 0 et 1)
*  qui nous donnent la matrice C1_data. 
*  La deuxième couche est une étape de sous-échantillonage par 2 qui va, sur chaque feature map,
*  moyenner chaque carré de pixels de taille de 2*2. Cela nous donne 6 matrices de taille 14*14, la 
*  taille des feature maps étant divisée par deux, et nous stockons les 6 matrices
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
#include <time.h>
#define BLOCK_SIZE 32


// ------------------------------- Fonctions d'initialisation de matrices ----------------------------

/* Ces premières fonctions nous aident à intialiser les matrices raw_data, C1_data, C1_kernel, et 
*  S1_data, et à les afficher avec MatrixPrint. Chaque fonction MatrixInit prend en argument un tableau 
*  (matrice) M à initialiser et le nombre total d'éléments qu'elle contient, appelé "size".
*  La fonction Matrix Print quant à elle prend une matrice M, et ses dimensions nx,ny,nz.
*/



/* MatrixInitInt : Initialise chaque valeur M[i] d'un tableau M par la valeur i */
void MatrixInitInt(float *M,int size)
{
    for(int i=0;i<size;i++){
        M[i]=i;
    }
}


/* MatrixInitOne : Initialise chaque valeur M[i] d'un tableau M par la valeur 1 */
void MatrixInitOne(float *M,int size)
{
    for(int i=0;i<size;i++){
        M[i]=1;
    }
}


/* MatrixInitZero : Initialise chaque valeur M[i] d'un tableau M par la valeur 0 */
void MatrixInitZero(float *M,int size)
{
    for(int i=0;i<size;i++){
        M[i]=0;
    }
}

/* MatrixInitRand : Initialise chaque valeur M[i] d'un tableau M par une valeur random entre 0 et 1
*   L'appel à la fonction rand renvoit un nombre entier positif choisi aléatoirement.
*   En ajoutant %1000, on obtient le reste de sa division euclidienne par 1000, soit un entier
*   positif aléatoirement compris entre 0 et 999.
*   L'élément M[i] étant le nombre flottant résultant de la division du nombre précédent par 1000,
*   il correspond donc à un nombre flottant aléatoirement compris entre 0 et 1 avec une précision de 10⁻3.
*   Si on avait voulu une précision de 10⁻4, on aurait remplacé les 10³ par des 10⁴.
*/
void MatrixInitRand(float *M, int size){
    for (int i = 0; i<size; i++){
        M[i] = (float)(rand()%1000)/1000 ; 
    }
}

/* MatrixPrint : Affiche une matrice les nz sous-matrices d'une matrice M
*
*    nx : nombre de lignes d'une sous-matrice
*    ny : nombre de colonnes d'une sous-matrice
*    nz : nombre de sous-matrices
*    
*    Exemple : pour afficher chacun des 6 kernels de taille 5*5, on a nx=5,ny=5,nz=6 car il y a 6 kernels
*    La fonction chacun des 6 kernels les uns en dessous des autres
*    
*    La fonction itère sur les nz sous-matrices l'affichage d'une sous-matrice, avec un saut de ligne à la fin de
*    l'affichage d'une sous-matrice. On affiche à chaque fois 3 chiffes avant la virgule et 1 chiffre (%3.1f). Si
*    Le nombre affiché est négatif, on affiche la valeur avec un espace après, si elle est positive, on ajoute un
*    espace au début pour qu'il n'y ait pas de problème de décalage d'affichage avec des valeurs négatives.
*/
void MatrixPrint(float *M,const int nx,const int ny,const int nz)
{
    printf("\nMatrix: (%d*%d*%d) \n",nx,ny,nz);
    for(int k=0;k<nz;k++){                                
        for(int i=0;i<ny;i++){                            
            for(int j=0;j<nx;j++){
                if(M[k*(nx*ny)+nx*i +j]<0){
                    printf("%3.1f ",M[k*(nx*ny)+nx*i +j]);                     
                }else{
                    printf(" %3.1f ",M[k*(nx*ny)+nx*i +j]);
                }
            }
            printf("\n");  // saut de ligne à la fin de l'affichage d'une sous-matrice

        }
        printf("\n");
    }
}

/* printthreadindex : Fonction qui sert à afficher l'indexage global d'un thread sur une grille, pour mieux comprendre 
*  l'indexage, a servi simplement de helper function pour comprendre l'indexage des valeurs des matrices. Sur matrice 
*  2D seulement, (dimensions nx*ny).
*/
__global__ void printthreadindex(float *M,const int nx,const int ny)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;  // ligne actuelle de la matrice M
    int iy=threadIdx.y+blockIdx.y*blockDim.y;  // colonne actuelle de la matrice M

    unsigned int idx=ix+iy*nx; // index global de l'élément M(ix,iy) dans la grille

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index %2d  ival %2d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,idx,M[idx]);

}


// --------------------------------- Convolution, sous-échantillonnage, et fonction d'activation ---------------------------


/* gpuMatrix2DConv : Fonction qui réaliser la convolution d'une matrice avec 1 SEUL kernel, et donne en sortie 1 seul feature map
*    Entree : Matrice d'entrée
*    Kernel : Matrice du kernel
*    Sortie : Matrice de sortie de la convolution
*    Ex : Nb de lignes de la matrice E
*    Ey : Nb de colonnes de la matrice E
*    kernel_size : Nb de lignes (ou de colonne comme un kernel est une matrice carrée ici) du kernel
*    Sx : Nb de lignes de la matrice S
*    Sy : Nb de colonnes de la matrice S 
*
*    Une fonction sur GPU étant exécutée sur UN thread, ici un thread correpond à un pixel de la matrice de Sortie, donc une convolution
*    du kernel avec 1 bloc kernel_size*kernel_size de l'image Entrée
*/

__global__ void gpuMatrix2DConv(float* Entree, float* Kernel, float* Sortie, int Ex, int Ey, int kernel_size, int Sx, int Sy)
{
    //Identifiants globaux ligne (row) et colonne (col) de la matrice Sortie
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Somme qui qui donnera à la fin la valeur du pixel S(row,col)
    float sum = 0.0;
    
    // si on est toujours à l'intérieur de la matrice E
    if (row < Sx && col < Sy) { 
        /* On itère sur les lignes et colonnes du kernel pour ajouter à sum la multiplication d'un pixel Kernel(i,j) avec le 
        *  pixel S(i,j) du bloc de taille kernel_size_*kernel_size considéré
        */
        for (int maskRow = 0; maskRow < kernel_size; maskRow++) {
            for (int maskCol = 0; maskCol < kernel_size; maskCol++) {
                sum += Entree[(row + maskRow) * Ey + (col + maskCol)] * Kernel[maskRow * kernel_size + maskCol];
            }
        }
        Sortie[row * Sy + col] = sum;
    }
}

/* activation_tanh : Fonction qui applique la fonction tanh à une valeur m 
*    C'est une fonction déclarée __device__ qui est exécutée par le device (GPU) et appelée par le device (GPU).
*    Elle doit être appellée dans un kernel et ne nécessite d'appel <<<B,T>>> comme les fonctions __global__
*    On l'appelle à la fin de la fonction gpuMatrix3DConv 
*/

__device__ float activation_tanh(float m){
    return tanhf(m);
}


/* gpuMatrix3DConv : Fonction qui réalise la convolution d'une matrice avec nb_kernels kernels, et donne en sortie nb_kernels 
*  feature maps dans la matrice de sortie. La matrice S sera de taille Sx*Sy*nb_kernels
*
*    Paramètres identiques à gpuMatrix2DConv, avec nb_kernels en plus : nombre de kernels avec lesquels on veut convoluer E 
*
*    On part de la fonction de convolution 2D pour construire la convolution 3D : à chaque thread, on calcule la valeur du pixel 
*    (i,j) de CHAQUE feature map. Un thread réalise donc nb_kernels convolutions avec un carré kernel_size*kernel_size de l'image .
*    On applique à chaque somme la fonction d'activation tanh.
*/

__global__ void gpuMatrix3DConv(float* Entree, float* Kernel, float* Sortie, int Ex, int Ey, int kernel_size, int nb_kernels, int Sx, int Sy){
    
    //Identifiants globaux ligne (row) et colonne (col) du thread actuel
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < Sx && col < Sy) {
        
        // On itère l'opération de convolution d'un carré kernel_size*kernel_size de la matrice d'entrée E sur chacun des nb_kernels kernels 
        for(int num_kernel = 0; num_kernel<nb_kernels; num_kernel++){
            
            // L'offset ici correpond au numéro de kernel de la matrice Kernel avec lequel on fait une opération de convolution sur E
            int offset = num_kernel*kernel_size*kernel_size;
            
            //On initialise la somme qui donnera la valeur finale du pixel (row,col) du feature map n° num_kernel de la matrice Sortie
            float sum = 0.0;
            
            for (int maskRow = 0; maskRow < kernel_size; maskRow++) {
                for (int maskCol = 0; maskCol < kernel_size; maskCol++) {
                    sum += Entree[(row + maskRow) * Ey + (col + maskCol) ] * Kernel[maskRow * kernel_size + maskCol + offset];
                }
            }
            Sortie[num_kernel*(Sx*Sy) + row * Sy + col] = activation_tanh(sum);
        
        }
    }
    
}

/* gpuMatrix3DConv_sans_tanh : fonction gpuMatrix3DConv sans la fonction tanh
*/

__global__ void gpuMatrix3DConv_sans_tanh(float* Entree, float* Kernel, float* Sortie, int Ex, int Ey, int kernel_size, int nb_kernels, int Sx, int Sy){
    
    //Identifiants globaux ligne (row) et colonne (col) du thread actuel
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < Sx && col < Sy) {
        
        // On itère l'opération de convolution d'un carré kernel_size*kernel_size de la matrice d'entrée E sur chacun des nb_kernels kernels 
        for(int num_kernel = 0; num_kernel<nb_kernels; num_kernel++){
            
            // L'offset ici correpond au numéro de kernel de la matrice Kernel avec lequel on fait une opération de convolution sur E
            int offset = num_kernel*kernel_size*kernel_size;
            
            //On initialise la somme qui donnera la valeur finale du pixel (row,col) du feature map n° num_kernel de la matrice Sortie
            float sum = 0.0;
            
            for (int maskRow = 0; maskRow < kernel_size; maskRow++) {
                for (int maskCol = 0; maskCol < kernel_size; maskCol++) {
                    sum += Entree[(row + maskRow) * Ey + (col + maskCol) ] * Kernel[maskRow * kernel_size + maskCol + offset];
                }
            }
            Sortie[num_kernel*(Sx*Sy) + row * Sy + col] = sum;
        
        }
    }
    
}


/* cudaMoyen2: fonction moyenneur, executée sur GPU
*
*    n : taille d'une ligne (et aussi d'une colonne) de la matrice d'entrée E (qui est carrée)
*
*    L'argument de la fonction correspond à la dimension de la matrice d'entrée
*    Les nombres de blocks et threads sont ceux de la matrice d'arrivés car le nombre de calculs corespond au nombre 
*    d'éléments à l'arrivée
*/

__global__ void cudaMoyen2(float *E, float *S, int n){
    
    // n_out : dimension de la matrice de sortie
    int n_out = n/2; 
    
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
    S[shift_S + output_row * n_out + output_col] = (float)(( E[shift_E + input_row * n + input_col] 
    + E[shift_E + (input_row+1) * n + input_col] + E[shift_E + input_row * n + (input_col+1)] + 
    E[shift_E + (input_row+1) * n + (input_col+1)] )/4);
    
}


// ********************************************* Main ***************************************************** 

/* Le main est la fonction principale dans laquelle nous effectuons les différentes opérations sur les
*  matrices. Nous effectuons plusieurs tests pour montrer le bon fonctionnement de nos fonctions.
* 
*/


int main()
{

// --------------------------------------- 1ère démonstration ---------------------------------------------
    
    /* Le premier test est réalisé sur une matrice raw_data de taille 8*8 composées uniquement de 1, et une 
    *  matrice C1_kernel composée de 6 kernels de taille 3*3, initialisés avec la matrice MatrixIntInit.
    *  Après convolution, la matrice C1_data est composées de 6 matrices de taille 6*6 (6 = 8-3+1), et un pixel
    *  du feature map n°k contient la somme des kernel(i,j) du kernel n°k. Par exemple, le premier kernel a ses 
    *  valeurs incrémentées de 0 à 8, et bien le premier feature map de C1_data a des 0+1+2+3+4+5+6+7+8=36 partout.
    *  Le sous-échantillonnage sans tanh donne ensuite une matrice S1_data de 6 matrices de taille 3*3 qui,
    *  comme les valeurs d'un feature map sont les mêmes partout, donnent des matrices de taille 3*3 avec les mêmes
    *  valeurs que dans leur feature map respectif (par exemple pour le feature map 1, on a à chaque pixel (36+36+36
    *  +36)/4 = 36 => la valeur ne change pas, le feature map sera ses dimensions divisées par 2. Avec le tanh, les
    *  valeurs du feature map sont compressées entre 0 et 1, et comme les valeurs dans les features maps sous_échanti-
    *  llonnées sont très grandes, les valeurs de S1_data seront toutes à 1.
    */
    
    printf("\n************ Test 1 ************\n");
    
    
    // ---------------------- Layer 1 : Initialisation -------------------------------
    
    /* Matrix raw_data de taille 8*8 avec que des 1*/
    int raw_size=8;
    float rawBytes=raw_size*raw_size*sizeof(float);
    
    // Allocation de mémoire pour les matrices sur CPU
    float *raw_data;
    raw_data=(float *)malloc(rawBytes);
    
    // Initialisation et printf
    MatrixInitOne(raw_data,raw_size*raw_size);
    printf("\nMatrice raw_data à 1\n");
    MatrixPrint(raw_data,raw_size,raw_size,1);
    
    // Allocation de mémoire sur GPU
    float *d_raw_data;
    cudaMalloc((void **)&d_raw_data,rawBytes);
    
    // Envoi de la matrice sur le device pour calculs sur GPU
    cudaMemcpy(d_raw_data,raw_data,rawBytes,cudaMemcpyHostToDevice);
    
    
    /*Matrix C1_Kernel de 6 kernels de taille 3*3*/ 
    int C1_kernel_size=3,nb_kernels=6;
    float C1_kernelBytes=C1_kernel_size*C1_kernel_size*nb_kernels*sizeof(float);
    
    float *C1_kernel;
    C1_kernel=(float *)malloc(C1_kernelBytes);
    
    MatrixInitInt(C1_kernel,C1_kernel_size*C1_kernel_size*nb_kernels);
    printf("\nMatrice C1_kernel initialisée avec des int\n");
    MatrixPrint(C1_kernel,C1_kernel_size,C1_kernel_size,nb_kernels);
    
    float *d_C1_kernel;
    cudaMalloc((void **)&d_C1_kernel,C1_kernelBytes);

    cudaMemcpy(d_C1_kernel,C1_kernel,C1_kernelBytes,cudaMemcpyHostToDevice);
    
    
    /*Matrix C1_data en sortie de gpuMatrix3DConv, avec 6 feature maps de taille 6*6*/
    int C1_data_size=6,nb_of_maps=6;
    float C1_data_Bytes=C1_data_size*C1_data_size*nb_of_maps*sizeof(float);
    
    float *C1_data;
    C1_data=(float *)malloc(C1_data_Bytes);

    float *d_C1_data;
    cudaMalloc((void **)&d_C1_data,C1_data_Bytes);

    MatrixInitZero(C1_data,C1_data_size*C1_data_size*nb_of_maps);
    //MatrixPrint(C1_data,C1_data_size,C1_data_size,nb_of_maps);
    
    cudaMemcpy(d_C1_data,C1_data,C1_data_Bytes,cudaMemcpyHostToDevice);
    
    /*Matrix S1_data en sortie de cudaMoyen2*/
    int S1_data_size=3;
    float S1_data_Bytes=S1_data_size*S1_data_size*nb_of_maps*sizeof(float);
    
    float *S1_data;
    S1_data=(float *)malloc(S1_data_Bytes);

    float *d_S1_data;
    cudaMalloc((void **)&d_S1_data,S1_data_Bytes);

    MatrixInitZero(S1_data,S1_data_size*S1_data_size*nb_of_maps);
    //MatrixPrint(S1_data,S1_data_size,S1_data_size,nb_of_maps);
    
    cudaMemcpy(d_S1_data,S1_data,S1_data_Bytes,cudaMemcpyHostToDevice);

    
    // -------------------- Layer 2 : Convolution 1 ----------------------------
    
    /* On a maximum 32*32 threads par bloc, et on a besoin de C1_data_size*C1_data_size/1024 blocs*/
    int threadsPerBlock = 32; 
    int gridCols = ceil(double(C1_data_size) / double(threadsPerBlock)); 
    int gridRows = ceil(double(C1_data_size) / double(threadsPerBlock));

    dim3 gridDim(gridCols, gridRows);
    dim3 blockDim(threadsPerBlock, threadsPerBlock);    // total 32x32=1024 threads
    //gpuMatrix2DConv << < gridDim, blockDim >> > (d_raw_data, d_C1_kernel, d_C1_data, raw_size, raw_size, C1_kernel_size, C1_data_size, C1_data_size);
    
    // Calcul de la convolution, ici sans tanh. Décommenter la ligne du dessous pour l'avoir avec tanh
    
    gpuMatrix3DConv_sans_tanh << < gridDim, blockDim >> > (d_raw_data, d_C1_kernel, d_C1_data, raw_size, raw_size, C1_kernel_size,nb_kernels, C1_data_size, C1_data_size);
    // gpuMatrix3DConv << < gridDim, blockDim >> > (d_raw_data, d_C1_kernel, d_C1_data, raw_size, raw_size, C1_kernel_size,nb_kernels, C1_data_size, C1_data_size);
    
    // ------------------ Layer 3 : Sous-échantillonage 1 ----------------------
    
    dim3 my_blocks (nb_of_maps, S1_data_size, 1); // = (6,3,1)
    cudaMoyen2<<<my_blocks,S1_data_size>>>(d_C1_data,d_S1_data, C1_data_size);
    
    
    // ----------------------Retour au CPU--------------------------------------
    
    // Récupération des données sur le CPU
    cudaMemcpy(C1_data, d_C1_data, C1_data_Bytes, cudaMemcpyDeviceToHost); // C1_data
    cudaMemcpy(S1_data, d_S1_data, S1_data_Bytes, cudaMemcpyDeviceToHost); // S1_data
    
    // Affichage de C1_data et S1_data 
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
    
    
// --------------------------------------- 2ème démonstration ---------------------------------------------
    
    /* Le 2ème test est réalisé selon les consignes du TP sur une matrice raw_data de taille 32*328 composée
    *  valeurs random entre 0 et 1, et une matrice C1_kernel composée de 6 kernels de taille 5*5, initialisés 
    *  également avec des valeurs entre 0 et 1.
    *  Après convolution, la matrice C1_data est composées de 6 matrices de taille 28*28, et le 
    *  sous-échantillonnage avec tanh donne ensuite une matrice S1_data de 6 matrices de taille 14*14 qui ont des
    *  valeurs entre 0 et 1 à cause du tanh.
    */
    
     printf("\n\n\n************ Test 2 ************\n");
    
    
    // ---------------------- Layer 1 : Initialisation -------------------------------
    
    /* Matrix raw_data de taille 32*32 avec que des valeurs random*/
    raw_size=32;
    rawBytes=raw_size*raw_size*sizeof(float);
    
    // Allocation de mémoire pour les matrices sur CPU
    raw_data=(float *)malloc(rawBytes);
    
    // Initialisation et printf
    MatrixInitRand(raw_data,raw_size*raw_size);
    printf("\nMatrice raw_data avec valeurs randoms entre 0 et 1\n");
    MatrixPrint(raw_data,raw_size,raw_size,1);
    
    // Allocation de mémoire sur GPU
    cudaMalloc((void **)&d_raw_data,rawBytes);
    
    // Envoi de la matrice sur le device pour calculs sur GPU
    cudaMemcpy(d_raw_data,raw_data,rawBytes,cudaMemcpyHostToDevice);
    
    
    /*Matrix C1_Kernel de 6 kernels de taille 5*5 */ 
    C1_kernel_size=5,nb_kernels=6;
    C1_kernelBytes=C1_kernel_size*C1_kernel_size*nb_kernels*sizeof(float);
    
    C1_kernel=(float *)malloc(C1_kernelBytes);
    
    MatrixInitRand(C1_kernel,C1_kernel_size*C1_kernel_size*nb_kernels);
    printf("\nMatrice C1_kernel avec valeurs randoms entre 0 et 1\n");
    MatrixPrint(C1_kernel,C1_kernel_size,C1_kernel_size,nb_kernels);
    
    cudaMalloc((void **)&d_C1_kernel,C1_kernelBytes);

    cudaMemcpy(d_C1_kernel,C1_kernel,C1_kernelBytes,cudaMemcpyHostToDevice);
    
    
    /*Matrix C1_data en sortie de gpuMatrix3DConv, avec 6 feature maps de taille 6*6*/
    C1_data_size=28,nb_of_maps=6;
    C1_data_Bytes=C1_data_size*C1_data_size*nb_of_maps*sizeof(float);
    
    C1_data=(float *)malloc(C1_data_Bytes);

    cudaMalloc((void **)&d_C1_data,C1_data_Bytes);

    MatrixInitZero(C1_data,C1_data_size*C1_data_size*nb_of_maps);
    //MatrixPrint(C1_data,C1_data_size,C1_data_size,nb_of_maps);
    
    cudaMemcpy(d_C1_data,C1_data,C1_data_Bytes,cudaMemcpyHostToDevice);
    
    /*Matrix S1_data en sortie de cudaMoyen2*/
    S1_data_size=14;
    S1_data_Bytes=S1_data_size*S1_data_size*nb_of_maps*sizeof(float);
    
    S1_data=(float *)malloc(S1_data_Bytes);

    cudaMalloc((void **)&d_S1_data,S1_data_Bytes);

    MatrixInitZero(S1_data,S1_data_size*S1_data_size*nb_of_maps);
    //MatrixPrint(S1_data,S1_data_size,S1_data_size,nb_of_maps);
    
    cudaMemcpy(d_S1_data,S1_data,S1_data_Bytes,cudaMemcpyHostToDevice);
    

    
    // -------------------- Layer 2 : Convolution 1 ----------------------------
    
    /* On a maximum 32*32 threads par bloc, et on a besoin de C1_data_size*C1_data_size/1024 blocs*/
    threadsPerBlock = 32; 
    gridCols = ceil(double(C1_data_size) / double(threadsPerBlock)); 
    gridRows = ceil(double(C1_data_size) / double(threadsPerBlock));

    dim3 gridDim2(gridCols, gridRows);
    dim3 blockDim2(threadsPerBlock, threadsPerBlock);    // total 32x32=1024 threads
    //gpuMatrix2DConv << < gridDim, blockDim >> > (d_raw_data, d_C1_kernel, d_C1_data, raw_size, raw_size, C1_kernel_size, C1_data_size, C1_data_size);
    gpuMatrix3DConv << < gridDim2, blockDim2 >> > (d_raw_data, d_C1_kernel, d_C1_data, raw_size, raw_size, C1_kernel_size,nb_kernels, C1_data_size, C1_data_size);
    
    // ------------------ Layer 3 : Sous-échantillonage 1 ----------------------
    
    dim3 my_blocks2(nb_of_maps, S1_data_size, 1); // = (6,3,1)
    
    /* Calcul de la matrice S1_data */
    cudaMoyen2<<<my_blocks2,S1_data_size>>>(d_C1_data,d_S1_data, C1_data_size);
    
    // On n'obtient que des 1 ! C'est parce que la plupart des valeurs en sortie de la convolution 
    //sont au-dessus de 1, et la fonction tanh vient les saturer toutes à 1.
    
    // ----------------------Retour au CPU--------------------------------------
    
    // Récupération des données sur le CPU
    cudaMemcpy(C1_data, d_C1_data, C1_data_Bytes, cudaMemcpyDeviceToHost); // C1_data
    cudaMemcpy(S1_data, d_S1_data, S1_data_Bytes, cudaMemcpyDeviceToHost); // S1_data
    
    // Affichage de C1_data et S1_data 
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
