### GPU加速（更新至2022.1.3）

#### 环境cuda11.2，vs stdio 2019 community

### 采用vs编译需要修改的地方

点击本项目选择**属性-链接器-系统**，**对其中的堆保留大小，堆提交大小，堆栈保留大小，堆栈提交大小修改为200000000**

点击本项目选择**属性-CUDA C/C++ 设置rdc为true** ，**在属性-链接器-输入将依赖项cudadevrt.lib 添加至附属依赖项**，以开启动态并行。


### 除了用VS提供的编译环境进行测试外，我们还采用了nvprof工具对gpu内核进行分析
### 采用nvcc编译事项

进入cmd项目同一路径下

```c
C:\Users\17628\Desktop\cuda_demo_double>
```

输入命令：（kernel.cu为执行文件，demo.o为生成）

**-arch=sm_50 -rdc=true** 为开启动态并行

```c
nvcc -arch=sm_50 -rdc=true kernel.cu -o demo.o 
```

采用命令，利用nvprof分析gpu性能

```c
nvprof ./demo.o
```

可以查看kernel在gpu上的运行情况

![cuda double 运行分析](cuda_speedup/cuda/cuda%20double%20运行分析.png)

### 同时nvidia官方有提供一个更加方便的可视化工具 **visual profiler**

通过在命令行中输入如下命令打开：
```c
nvvp
```
### 在初始的界面中选择file-new session，将之前生成的demo.o文件载入，**此时为double类型数据情况**

![nvprof可视化载入](cuda_speedup/cuda/nvprof可视化载入.png)

此时进入分析界面，界面中显示了gpu在运行中的各种情况说明：

**并行求和运行情况**
![sumArray运行可视化](cuda_speedup/cuda/sumArray运行可视化.png)

**并行求最大值运行情况**
![maxArray运行可视化](cuda_speedup/cuda/maxArray运行可视化.png)

**花费总时间**

![cuda double 运行总时间](cuda_speedup/cuda/cuda%20double%20运行总时间.png)


### 在visual profiler中载入**float数据类型**的情况下：

**并行求和运行情况**
![sumArray运行可视化 float](cuda_speedup/cuda/sumArray运行可视化%20float.png)

**并行求最大值运行情况**
![maxArray运行可视化 float](cuda_speedup/cuda/maxArray运行可视化%20float.png)

**花费总时间**

![cuda float 运行总时间](cuda_speedup/cuda/cuda%20float%20运行总时间.png)


### 以上是两种数据类型下sum和max并行执行的概述

#### 下面来看每个部分中gpu中的运行情况分析

#### 比较sum中数据类型导致的差异

#### 1、比较sum中cudaMalloc函数

**cudaMalloc函数（申请GPU内存）在double情况下运行时间217.8929ms**

![cuda double sum运行分析 cudaMalloc](cuda_speedup/cuda/cuda%20double%20sum运行分析%20cudaMalloc.png)

![cuda double sum运行分析 cudaMalloc2](cuda_speedup/cuda/cuda%20double%20sum运行分析%20cudaMalloc2.png)

**cudaMalloc函数（申请GPU内存）在float情况下运行时间223.7232ms**

![cuda flaot sum运行分析 cudaMalloc](cuda_speedup/cuda/cuda%20float%20sum运行分析%20cudaMalloc.png)

#### 可见在申请GPU内存中，两种数据类型并未造成巨大影响

#### 2、比较sum中cudaMemcpy函数

**cudaMemcpy函数（CPU向GPU传输数据）在double情况下运行时间217.2695ms**

![cuda double sum运行分析 cudaMemcpy](cuda_speedup/cuda/cuda%20double%20sum运行分析%20cudaMemcpy.png)

**cudaMemcpy函数（CPU向GPU传输数据）在flaot情况下运行时间108.0027ms**

![cuda flaot sum运行分析 cudaMemcpy](cuda_speedup/cuda/cuda%20float%20sum运行分析%20cudaMemcpy.png)

#### 传输字节数double是float的两倍，所以造成了此处的时间有两倍的差距

#### 3、比较sum中SumArray函数

**SumArray函数（自行编写的求和核函数）在double情况下运行时间68.126ms**

![cuda double sum运行分析 SumArray](cuda_speedup/cuda/cuda%20double%20sum运行分析%20SumArray.png)

**SumArray函数（自行编写的求和核函数）在flaot情况下运行时间5.4655ms**

![cuda flaot sum运行分析 SumArray](cuda_speedup/cuda/cuda%20float%20sum运行分析%20SumArray.png)

#### 从数据中看到gpu在运行float和double类型运算时速度会相差10倍以上



