#include <cuda_runtime.h>
#include <stdio.h>



void MatrixInitInt(float *M,int size)
{
    for(int i=0;i<size;i++){
        M[i]=i;
    }

}

void MatrixInitOne(float *M,int size)
{
    for(int i=0;i<size;i++){
        M[i]=1;
    }

}
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

__global__ void printthreadindex(float *A,const int nx,const int ny)
{
    int ix=threadIdx.x+blockIdx.x*blockDim.x;
    int iy=threadIdx.y+blockIdx.y*blockDim.y;
    //int iz=threadIdx.z+blockIdx.z*blockDim.z;
    
    
    unsigned int idx=ix+iy*nx; //+nx*ny*iz;

    printf("thread_id (%d,%d) block_id (%d,%d) coordinate (%d,%d) global index %2d  ival %2d \n",threadIdx.x,threadIdx.y,blockIdx.x,blockIdx.y,ix,iy,idx,A[idx]);

}

__global__ void gpuMatrixConv(float* gpuMat1, float* kernel, float* gpuMat3, int m1Rows, int m1Cols, int mRowsCols, int m3Rows, int m3Cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0;

    if (row < m3Rows && col < m3Cols) {
        for (int maskRow = 0; maskRow < mRowsCols; maskRow++) {
            for (int maskCol = 0; maskCol < mRowsCols; maskCol++) {
                sum += gpuMat1[(row + maskRow) * m1Cols + col + maskCol] * kernel[maskRow * mRowsCols + maskCol];
            }
        }
        gpuMat3[row * m3Cols + col] = sum;
    }
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
    int C1_kernel_size=3; //,nz=3;
    float C1_kernelBytes=C1_kernel_size*C1_kernel_size*sizeof(float);
    
    float *C1_kernel;
    C1_kernel=(float *)malloc(C1_kernelBytes);
    
    MatrixInitOne(C1_kernel,C1_kernel_size*C1_kernel_size);
    MatrixPrint(C1_kernel,C1_kernel_size,C1_kernel_size,1);
    
    float *d_C1_kernel;
    cudaMalloc((void **)&d_C1_kernel,C1_kernelBytes);

    cudaMemcpy(d_C1_kernel,C1_kernel,C1_kernelBytes,cudaMemcpyHostToDevice);
    
    
    //Matrix C1_data output of convolution 1
    int C1_data_size=6; //,nz=3;
    float C1_data_Bytes=C1_data_size*C1_data_size*sizeof(float);
    
    float *C1_data;
    C1_data=(float *)malloc(C1_data_Bytes);

    float *d_C1_data;
    cudaMalloc((void **)&d_C1_data,C1_data_Bytes);

    MatrixInitZero(C1_data,C1_data_size*C1_data_size);
    MatrixPrint(C1_data,C1_data_size,C1_data_size,1);
    
    cudaMemcpy(d_C1_data,C1_data,C1_data_Bytes,cudaMemcpyHostToDevice);
    
    // Process
    //dim3 block(nx);//,3);
    //dim3 grid(ny);
    //printthreadindex <<<grid,block>>> (d_MatA,nx,ny);//,1);
    
    int threadsPerBlock = 32;

	int gridCols = ceil(double(6) / double(threadsPerBlock));
	int gridRows = ceil(double(6) / double(threadsPerBlock));

	dim3 gridDim(gridCols, gridRows);
	dim3 blockDim(threadsPerBlock, threadsPerBlock);	// total 32x32=1024 threads

	//nvcc
	gpuMatrixConv << < gridDim, blockDim >> > (d_raw_data, d_C1_kernel, d_C1_data, raw_size, raw_size, C1_kernel_size, C1_data_size, C1_data_size);
    
    cudaMemcpy(C1_data, d_C1_data, C1_data_Bytes, cudaMemcpyDeviceToHost);
    printf("\nConvolution\n");
    MatrixPrint(C1_data,C1_data_size,C1_data_size,1);

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