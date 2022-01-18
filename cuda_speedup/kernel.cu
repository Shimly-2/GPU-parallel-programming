
//环境:CUDA11.2，vs2019

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <stdio.h>
#include <iostream>
#include <time.h>
#include "omp.h" 

/*------------函数声明------------*/
double addWithCuda(float* divideData, float* rawFloatData);
double maxWithCuda(float* result_data, float* rawFloatData);
double sum(const float data[], const int len); //data是原始数据，len为长度。结果通过函数返回
double max1(const float data[], const int len);
bool is_check(const float data[], int len);
void initialize_data(float* rawFloatData, unsigned int len);
void sortWithCuda(float* rawFloatData, unsigned int len);
double sort2(const float data[], const int len, float result[]);
void quicksort(float data[], int start, int end);
void swap(float& a, float& b);
void initcuda();

/*------------相关线程块，数组宏定义------------*/
#define MAX_DEPTH        16
#define INSERTION_SORT   32
#define MAX_THREADS      64
#define SUBDATANUM       2000000
#define DATANUM          (SUBDATANUM * MAX_THREADS)   /*这个数值是总数据量*/
#define DATANUM2         640000                       /*这个数值是总数据量*/
#define BLOCKS_PerGrid   512
#define THREADS_PerBlock 1024 //2^8

/*------------check函数，检测cuda调用内存时的错误------------*/
#define checkCudaErrors( a ) {if (a != cudaSuccess) { \
                               fprintf(stderr, "cudaMalloc failed!");\
                               goto Error;}}


/*********************************************************************************
	* FunctionName:  SumArray
	* Description：  用于cuda并行求和的核函数
	* Calls:         none
	* Called By:     sumWithCuda
	* Input:         divideData[N]    返回线程块数组
					 rawFloatData[N]  待求和数组
	* Output:        divideData[N]    返回线程块数组
	* Return:        none
**********************************************************************************/
__global__ void SumArray(float* divideData, float* rawFloatData)//, int *b)
{
	__shared__ float mycache[THREADS_PerBlock];//设置每个块内同享内存threadsPerBlock==blockDim.x

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int bid = gridDim.x * blockDim.x;//每个grid里一共有多少个线程
	int cacheN;
	unsigned k;
	double sum;

	sum = 0.0;

	cacheN = threadIdx.x; //

	while (tid < DATANUM)
	{
		sum += log(sqrt(rawFloatData[tid]));
		tid = tid + bid;
	}

	mycache[cacheN] = sum;

	__syncthreads();//对线程块进行同步；等待该块里所有线程都计算结束
	//下面开始计算本block中每个线程得到的sum(保存在mycache）的和
	//1：线程对半加：

	k = THREADS_PerBlock >> 1;
	while (k)
	{
		if (cacheN < k)
		{
			//线程号小于一半的线程继续运行这里加
			mycache[cacheN] += mycache[cacheN + k];//数组序列对半加，得到结果，放到前半部分数组，为下次递归准备
		}
		__syncthreads();//对线程块进行同步；等待该块里所有线程都计算结束
		k = k >> 1;//数组序列，继续对半,准备后面的递归
	}
	//最后一次递归是在该块的线程0中进行，所有把线程0里的结果返回给CPU
	if (cacheN == 0)
	{
		divideData[blockIdx.x] = mycache[0];
	}
}


/*********************************************************************************
	* FunctionName:  MaxArray
	* Description：  用于cuda并行求最大值的核函数
	* Calls:         none
	* Called By:     maxWithCuda
	* Input:         divideData2[N]   返回线程块数组
					 rawFloatData[N]  待求和数组
	* Output:        divideData2[N]   返回线程块数组
	* Return:        none
**********************************************************************************/
__global__ void MaxArray(float rawFloatData[DATANUM], float divideData2[BLOCKS_PerGrid])
{
	__shared__ float mycache[THREADS_PerBlock];//设置每个块内同享内存threadsPerBlock==blockDim.x

	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int bid = gridDim.x * blockDim.x;//每个grid里一共有多少个线程
	int cacheN;
	unsigned k;
	double max;

	max = 0.0;

	cacheN = threadIdx.x; //

	while (tid < DATANUM)
	{

		if (log(sqrt(rawFloatData[tid])) > max)
		{
			max = log(sqrt(rawFloatData[tid]));
		}
		tid = tid + bid;
	}

	mycache[cacheN] = max;

	__syncthreads();//对线程块进行同步；等待该块里所有线程都计算结束
	//下面开始计算本block中每个线程得到的sum(保存在mycache）的和
	//1：线程对半加：

	k = THREADS_PerBlock >> 1;
	while (k)
	{
		if (cacheN < k)
		{
			//线程号小于一半的线程继续运行这里加
			if (mycache[cacheN + k] > mycache[cacheN])
			{
				mycache[cacheN] = mycache[cacheN + k];
			}
		}
		__syncthreads();//对线程块进行同步；等待该块里所有线程都计算结束
		k = k >> 1;//数组序列，继续对半,准备后面的递归
	}
	//最后一次递归是在该块的线程0中进行，所有把线程0里的结果返回给CPU
	if (cacheN == 0)
	{
		divideData2[blockIdx.x] = mycache[0];
	}
}


/*********************************************************************************
	* FunctionName:  selection_sort
	* Description：  用于cuda并行排序
	* Calls:         none
	* Called By:     cuda_quicksort
	* Input:         data[N]          排序数组
					 left             数组最左边下标
					 right            数组最右边下标
	* Output:        data[N]          排序数组
	* Return:        none
**********************************************************************************/
__device__ void selection_sort(float* data, int left, int right)
{
	for (int i = left; i <= right; ++i)
	{
		double min_val = log(sqrt(data[i]));
		int min_idx = i;

		// Find the smallest value in the range [left, right].
		for (int j = i + 1; j <= right; ++j)
		{
			double val_j = log(sqrt(data[j]));

			if (val_j < min_val)
			{
				min_idx = j;
				min_val = val_j;
			}
		}
		// Swap the values.
		if (i != min_idx)
		{
			data[min_idx] = log(sqrt(data[i]));
			data[i] = min_val;
		}
	}
}


/*********************************************************************************
	* FunctionName:  cuda_quicksort
	* Description：  用于cuda并行排序的核函数
	* Calls:         selection_sort，cuda_quicksort
	* Called By:     sortWithCuda，cuda_quicksort
	* Input:         data[N]          排序数组
					 left             数组最左边下标
					 right            数组最右边下标
					 depth            线程块深度
	* Output:        data[N]          排序数组
	* Return:        none
**********************************************************************************/
__global__ void cuda_quicksort(float* data, int left, int right, int depth)
{
	// If we're too deep or there are few elements left, we use an insertion sort...
	if (depth >= MAX_DEPTH || right - left <= INSERTION_SORT)
	{
		selection_sort(data, left, right);
		return;
	}

	float* lptr = data + left;
	float* rptr = data + right;
	float  pivot = data[(left + right) / 2];
	// Do the partitioning.
	while (lptr <= rptr)
	{
		// Find the next left- and right-hand values to swap
		float lval = *lptr;
		float rval = *rptr;
		// Move the left pointer as long as the pointed element is smaller than the pivot.
		while (lval < pivot)
		{
			lptr++;
			lval = *lptr;
		}
		// Move the right pointer as long as the pointed element is larger than the pivot.
		while (rval > pivot)
		{
			rptr--;
			rval = *rptr;
		}
		// If the swap points are valid, do the swap!
		if (lptr <= rptr)
		{
			*lptr++ = rval;
			*rptr-- = lval;
		}
	}
	// Now the recursive part
	float nright = rptr - data;
	float nleft = lptr - data;
	// Launch a new block to sort the left part.
	if (left < (rptr - data))
	{
		cudaStream_t s;
		cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
		cuda_quicksort << < 1, 1, 0, s >> > (data, left, nright, depth + 1);
		cudaStreamDestroy(s);
	}
	// Launch a new block to sort the right part.
	if ((lptr - data) < right)
	{
		cudaStream_t s1;
		cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
		cuda_quicksort << < 1, 1, 0, s1 >> > (data, nleft, right, depth + 1);
		cudaStreamDestroy(s1);
	}
}


int main()
{
	/*------------变量初始化------------*/
	std::cout << "Initializing data:" << std::endl;
	float* rawFloatData = new float[DATANUM];
	float* result = new float[DATANUM];
	float divideData[BLOCKS_PerGrid];
	float result_data[BLOCKS_PerGrid];
	unsigned int j;
	bool is_ok;
	LARGE_INTEGER start, end, fre;
	QueryPerformanceFrequency(&fre);

	/*------------cuda初始化------------*/
	initcuda();

	/*------------数组初始化------------*/
	initialize_data(rawFloatData, DATANUM);

	/*------------cuda并行求和------------*/
	QueryPerformanceCounter(&start);//start  
	double sum1 = addWithCuda(divideData, rawFloatData);
	/*------------所有线程块求和得到最终结果------------*/
	for (int j = 0; j < BLOCKS_PerGrid; j++)
	{
		sum1 += divideData[j];
	}
	QueryPerformanceCounter(&end);//end
	printf(" <cuda> sum=%lf   ", sum1);
	std::cout << "Time Consumed:" << (double)(end.QuadPart - start.QuadPart) * 1000 / (double)(fre.QuadPart) << "ms" << std::endl;

	/*------------用CPU验证求和是否正确------------*/
	//QueryPerformanceCounter(&start);//start 
	//double sum2 = sum(rawFloatData, DATANUM);
	//QueryPerformanceCounter(&end);//end
	//printf(" <cpu>  sum=%lf   ", sum2);
	//std::cout << "Time Consumed:" << (double)(end.QuadPart - start.QuadPart) * 1000 / (double)(fre.QuadPart) << "ms" << std::endl;

	/*------------cuda并行求最大值------------*/
	QueryPerformanceCounter(&start);//start  
	double max = maxWithCuda(result_data, rawFloatData);
	/*------------所有线程块求最大值得到最终结果------------*/
	for (int i = 0; i < BLOCKS_PerGrid; i++)
	{
		if (result_data[i] > max)
		{
			max = result_data[i];
		}
	}
	QueryPerformanceCounter(&end);//end
	printf(" <cuda> max=%lf  \t\t", max);
	std::cout << "Time Consumed:" << (double)(end.QuadPart - start.QuadPart) * 1000 / (double)(fre.QuadPart) << "ms" << std::endl;

	/*------------用CPU验证求最大值是否正确------------*/
	//QueryPerformanceCounter(&start);//start  
	//double max2 = max1(rawFloatData, DATANUM);
	//QueryPerformanceCounter(&end);//end
	//printf(" <cpu>  max=%lf  ", max2);
	//std::cout << "Time Consumed:" << (double)(end.QuadPart - start.QuadPart) * 1000 / (double)(fre.QuadPart) << "ms" << std::endl;

	/*------------cuda并行排序------------*/
	QueryPerformanceCounter(&start);//start  
	sortWithCuda(rawFloatData, DATANUM2);
	QueryPerformanceCounter(&end);//end
	is_ok = is_check(rawFloatData, DATANUM2);
	printf(" <cuda> sort result is %d  \t", is_ok);
	std::cout << "Time Consumed:" << (double)(end.QuadPart - start.QuadPart) * 1000 / (double)(fre.QuadPart) << "ms" << std::endl;

	/*------------cpu快速排序------------*/
	//QueryPerformanceCounter(&start);//start  
	////sort2(rawFloatData, DATANUM, result);
	//sort2(rawFloatData, DATANUM2, rawFloatData);
	//QueryPerformanceCounter(&end);//end
	////is_ok = is_check(result, DATANUM);
	//is_ok = is_check(rawFloatData, DATANUM2);
	//printf(" <cpu>  sort result is %d  ", is_ok);
	//std::cout << "Time Consumed:" << (double)(end.QuadPart - start.QuadPart) * 1000 / (double)(fre.QuadPart) << "ms" << std::endl;

	cudaError_t cudaStatus;
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	delete rawFloatData;
	return 0;
}

/*********************************************************************************
	* FunctionName:  addWithCuda
	* Description：  用于cuda并行求和
	* Calls:         SumArray
	* Called By:     main
	* Input:         divideData[N]    返回线程块数组
	                 rawFloatData[N]  待求和数组
	* Output:        divideData[N]    返回线程块数组
	* Return:        true/false
**********************************************************************************/
double addWithCuda(float* divideData, float* rawFloatData)
{
	float* dev_a = 0;
	float* dev_c = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));

	// 申请GPU内存空间
	checkCudaErrors(cudaMalloc((void**)& dev_a, DATANUM * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)& dev_c, BLOCKS_PerGrid * sizeof(float)));
	
	//将数据从cpu中复制到gpu中
	checkCudaErrors(cudaMemcpy(dev_a, rawFloatData, DATANUM * sizeof(float), cudaMemcpyHostToDevice));

	//启动GPU上的每个单元的线程
	SumArray << <BLOCKS_PerGrid, THREADS_PerBlock >> > (dev_c, dev_a);//, dev_b);

	//等待全部线程运行结束
	checkCudaErrors(cudaDeviceSynchronize());

	//将数据从gpu复制到cpu中
	checkCudaErrors(cudaMemcpy(divideData, dev_c, BLOCKS_PerGrid * sizeof(float), cudaMemcpyDeviceToHost));

Error:
	cudaFree(dev_c);
	cudaFree(dev_a);

	return 0.0;
}


/*********************************************************************************
	* FunctionName:  maxWithCuda
	* Description：  用于cuda并行求最大值
	* Calls:         MaxArray
	* Called By:     main
	* Input:         result_data[N]   返回线程块数组
	                 rawFloatData[N]  待求和数组
	* Output:        result_data[N]   返回线程块数组
	* Return:        true/false
**********************************************************************************/
double maxWithCuda(float* result_data, float* rawFloatData)
{
	float* rawfloatData;
	float* divideData2;

	// Choose which GPU to run on, change this on a multi-GPU system.
	checkCudaErrors(cudaSetDevice(0));

	// 申请GPU内存空间
	checkCudaErrors(cudaMalloc((void**)& rawfloatData, DATANUM * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)& divideData2, BLOCKS_PerGrid * sizeof(float)));

	//将数据从cpu中复制到gpu中
	checkCudaErrors(cudaMemcpy(rawfloatData, rawFloatData, DATANUM * sizeof(float), cudaMemcpyHostToDevice));

	//启动GPU上的每个单元的线程
	MaxArray << <BLOCKS_PerGrid, THREADS_PerBlock >> > (rawfloatData, divideData2); //调用内核函数

	//等待全部线程运行结束
	checkCudaErrors(cudaDeviceSynchronize());

	// Copy output vector from GPU buffer to host memory.
	checkCudaErrors(cudaMemcpy(result_data, divideData2, sizeof(float) * BLOCKS_PerGrid, cudaMemcpyDeviceToHost));

Error:
	cudaFree(rawfloatData);
	cudaFree(divideData2);

	return 0;
}


/*********************************************************************************
	* FunctionName:  sortWithCuda
	* Description：  用于cuda并行排序
	* Calls:         cdp_simple_quicksort
	* Called By:     main
	* Input:         rawFloatData[N]  排序数组
	                 len              数组长度
	* Output:        none
	* Return:        none       
**********************************************************************************/
void sortWithCuda(float* rawFloatData, unsigned int len)
{
	// Prepare CDP for the max depth 'MAX_DEPTH'.
	cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, MAX_DEPTH);
	cudaError_t cudaStatus;
	float* d_data = 0;
	// Launch on device
	int left = 0;
	int right = len - 1;

	for (int i = 0; i < len; i++)
	{
		rawFloatData[i] = log(sqrt(rawFloatData[i]));
	}

	checkCudaErrors(cudaMalloc((void**)& d_data, DATANUM * sizeof(float)));
	checkCudaErrors(cudaMemcpy(d_data, rawFloatData, DATANUM * sizeof(float), cudaMemcpyHostToDevice));

	//std::cout << "Launching kernel on the GPU" << std::endl;
	cuda_quicksort << < 1, 1 >> > (d_data, left, right, 0);
	checkCudaErrors(cudaDeviceSynchronize());

	cudaMemcpy(rawFloatData, d_data, DATANUM * sizeof(float), cudaMemcpyDeviceToHost);

Error:
	cudaFree(d_data);
}


/*********************************************************************************
	* FunctionName:  sum
	* Description：  用于cpu串行求和
	* Calls:         none
	* Called By:     main
	* Input:         data[N]          待求和数组
	                 len              数组长度
	* Output:        result           求和结果
	* Return:        result
**********************************************************************************/
double sum(const float data[], const int len)
{
	double result = 0;
	for (int i = 0; i < len; i++)
	{
		result += log(sqrt(data[i]));
	}
	return result;
}


/*********************************************************************************
	* FunctionName:  max1
	* Description：  用于cpu串行求最大值
	* Calls:         none
	* Called By:     main
	* Input:         data[N]          待求数组
	                 len              数组长度
	* Output:        result           求最大值结果
	* Return:        result
**********************************************************************************/
double max1(const float data[], const int len)
{
	double result = 0;
	for (int i = 0; i < len; i++)
	{
		if (log(sqrt(data[i])) > result)
		{
			result = log(sqrt(data[i]));
		}
	}
	return result;
}


/*********************************************************************************
	* FunctionName:  initcuda
	* Description：  用于cuda初始化
	* Calls:         none
	* Called By:     main
	* Input:         none
	* Output:        gpu              相关性能参数
	* Return:        none
**********************************************************************************/
void initcuda()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);
	int dev;
	for (dev = 0; dev < deviceCount; dev++)
	{
		int driver_version(0), runtime_version(0);
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);
		if (dev == 0)
			if (deviceProp.minor = 9999 && deviceProp.major == 9999)
				printf("\n");
		//显卡型号
		printf(" GPU Device : %s\n", deviceProp.name);
		//显卡驱动的版本号
		cudaDriverGetVersion(&driver_version);
		printf(" CUDA Driver Version:                            %d.%d\n",
			driver_version / 1000, (driver_version % 1000) / 10);
		//CUDA toolkit版本号
		cudaRuntimeGetVersion(&runtime_version);
		printf(" CUDA Runtime Version:                           %d.%d\n",
			runtime_version / 1000, (runtime_version % 1000) / 10);
		//显卡的计算能力（Compute Capability）
		printf(" Device Prop:                                    %d.%d\n",
			deviceProp.major, deviceProp.minor);
		//显存大小
		printf(" Total amount of Global Memory:                  %u bytes\n",
			deviceProp.totalGlobalMem);
		//SMX数量
		printf(" Number of SMs:                                  %d\n",
			deviceProp.multiProcessorCount);
		printf(" Total amount of Constant Memory:                %u bytes\n",
			deviceProp.totalConstMem);
		//一个block的共享内存大小
		printf(" Total amount of Shared Memory per block:        %u bytes\n",
			deviceProp.sharedMemPerBlock);
		printf(" Total number of registers available per block:  %d\n",
			deviceProp.regsPerBlock);
		printf(" Warp size:                                      %d\n",
			deviceProp.warpSize);
		printf(" Maximum number of threads per SM:               %d\n",
			deviceProp.maxThreadsPerMultiProcessor);
		//block最大线程数
		printf(" Maximum number of threads per block:            %d\n",
			deviceProp.maxThreadsPerBlock);
		//线程块三维
		printf(" Maximum size of each dimension of a block:      %d x %d x %d\n",
			deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
		//线程格三维
		printf(" Maximum size of each dimension of a grid:       %d x %d x %d\n",
			deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
		printf(" Maximum memory pitch:                           %u bytes\n",
			deviceProp.memPitch);
		printf(" Texture alignmemt:                              %u bytes\n",
			deviceProp.texturePitchAlignment);
		//时钟频率
		printf(" Clock rate:                                     %.2f GHz\n",
			deviceProp.clockRate * 1e-6f);
		printf(" Memory Clock rate:                              %.0f MHz\n",
			deviceProp.memoryClockRate * 1e-3f);
		printf(" Memory Bus Width:                               %d-bit\n\n",
			deviceProp.memoryBusWidth);
	}
}


/*********************************************************************************
	* FunctionName:  is_check
	* Description：  用于验证数组的递增性
	* Calls:         none
	* Called By:     main
	* Input:         data[N]          待求数组
	                 len              数组长度
	* Output:        is_ok            数组是否递增
	* Return:        is_ok
**********************************************************************************/
bool is_check(const float data[], int len)
{
	bool is_ok = true;
	for (int i = 0; i < len - 1; i++)
	{
		if (data[i + 1] - data[i] < 0)
		{
			is_ok = false;
			std::cout << "Invalid item[" << i - 1 << "]: " << data[i - 1] << " greater than " << data[i] << std::endl;
			exit(EXIT_FAILURE);
		}
	}
	return is_ok;
}

/*********************************************************************************
	* FunctionName:  initialize_data
	* Description：  用于数组初始化赋值
	* Calls:         none
	* Called By:     main
	* Input:         rawFloatData[N]  待求数组
	                 len              数组长度
	* Output:        rawFloatData[N]  待求数组
	* Return:        none
**********************************************************************************/
void initialize_data(float* rawFloatData, unsigned int len)
{
	// Fill dst with random values
	for (unsigned i = 0; i < len; i++)
		rawFloatData[i] = float(i + 1);
}


/*********************************************************************************
	* FunctionName:  quicksort
	* Description：  用于cpu数组快速排序递归
	* Calls:         quicksort
	* Called By:     quicksort，sort2
	* Input:         Data[N]          待排序数组
	                 start            数组开始下标
					 end              数组结束下标
	* Output:        data[N]          待求数组
	* Return:        none
**********************************************************************************/
void quicksort(float data[], int start, int end)
{
	int m;
	if (start >= end) return;
	m = start;
	for (int i = start + 1; i <= end; i++)
		if (data[i] < data[start])
			swap(data[++m], data[i]);
	swap(data[start], data[m]);
	quicksort(data, start, m - 1);
	quicksort(data, m + 1, end);
}



/*********************************************************************************
	* FunctionName:  sort2
	* Description：  用于cpu数组排序提供规范接口
	* Calls:         quicksort
	* Called By:     main
	* Input:         Data[N]          待排序数组
					 len              数组长度
					 result[N]        排序后数组
	* Output:        result[N]        排序后数组
	* Return:        true/false
**********************************************************************************/
double sort2(const float data[], const int len, float result[])
{
	int start = 0, end = len - 1;
	//for (int i = 0; i < len; i++)
	//{
	//	//result[i] = log(sqrt(data[i]));
	//	result[i] = data[i];
	//}
	quicksort(result, start, end);
	return true;
}


void swap(float& a, float& b)
{
	float tmp;
	tmp = a;
	a = b;
	b = tmp;
}