# ConvBench: A Comprehensive Convolution Performance Assessment Benchmark

ConvBench provides an end-to-end platform for evaluating convolution algorithms.
Drawing insights from the literature on convolution algorithms, we've defined a standardized nomenclature for the various processing steps that are inherent in these algorithms. Furthermore, we linked the standardized nomenclature with a  timing assessment API that ensures a fair environment for comparing different approaches across more than 9 thousand different 2D convolution layers extracted from all TIMM's deep learning model collections.

> [!TIP] 
> **TL;DR** - Explore the [ConvBench.ipynb](ConvBench.ipynb) Jupyter notebook for a simple all-in-one interface for ConvBench. It takes care of all filtering, compiling, and plotting particularities. Just modify the convolution and convolution_baseline cells to start using the benchmark!

## Detailed Project Description

### Convolution Operation Set

This step refers to the Convolution set construction.
For each available model from TIMM’s deep learning model collection (the largest DL model collection on the internet), we iterated over its layers and extracted information about all encountered 2D convolution operations.
In the end, we acquired 9011 different convolution operations, among which:

- 5481 are elementwise operations.
- Of the remainder 3530 convolutions:
    - 2269 are grouped convolutions;
	- 17 are dilated convolutions;
	- 93 have rectangular filters;
	- 2 expect rectangular inputs and have rectangular filters;
	- Only 1156 are related to common convolutions.

> [!TIP]
> For more information about these different kinds of convolutions, refer to [Convolution GIFs](https://github.com/vdumoulin/conv_arithmetic).

The hyperparameters of the 1156 common convolutions have the following data distributions:

![hp distribution](utils/size_and_channels.png)

Meanwhile, the number of Float Point Operations (FLOPS) for each common convolution is distributed in the range of 0.0009 GFLOPS and 43.49 GFLOPS, covering both small and fast operations and large and computationally expensive ones

![FLOPS distribution](utils/gflops.png)

|       |      |      | GFLOPS |        |        |        |       |
|-------|------|------|--------|--------|--------|--------|-------|
| Count | Mean | STD  | Min    | 25%    | 50%    | 75%    | Max   |
| 1156  | 0.83 | 2.55 | 0.0009 | 0.0578 | 0.1971 | 0.5609 | 43.49 |


### ConvBench Library

The ConvBench Benchmark is designed as an encapsulation class that must be included, inherited, and overridden. Its main methods are:

- `virtual void convolution(...)`: The main convolution algorithm to be assessed (it must be overridden).
- `virtual void convolution_baseline(...)`: The baseline algorithm to be compared (it must be overridden).
- `void convset_exec(data_strategy, running_strategy)`: The execution procedure. It expects a data_strategy enum type, selecting how the data buffers will be filled (random, ~~constant~~, ~~follow_dist~~, or ~~load~~), and a running_strategy enum type selecting which conv algorithm will be called (direct, baseline, or correctness).

> [!IMPORTANT]
> Things you must pay attention to when implementing your convolution algorithm: 
> - ConvBench is a templated class by `<T>`, allowing the child classes to select the type of the `inputs`, `kernels`, `bias`, and `outputs` data buffers.
> - Within the scope of ConvBench, there are some typedefs that we encourage you to use, such as:
>     - dimension type (`dim_t`, an unsigned `int64`); 
>     - index type (`index_t`, a signed `int64`);
>     - shape type (`shape_t`, a vector of `dim_t` objects).
> - The random number generator follows a normal distribution by default.
> - Activate/Deactivate LOG and DEBUG directives during compilation time (-DCERR_LOG=0/1 and -DCERR_DEBUG=0/1).
> - Remember to write a simple main function that instantiates the child class and calls the convset_exec method.

### Timing class

![timing_nomenclature](utils/timing_nomenclature.png)

Based on the literature and DL frameworks like oneDNN, ONNX-MLIR, GLOW, and PyTorch, a convolution operation can be broken down into different steps.

1. **Pre-Convolution Packing**: The data reordering preprocessing step to transform the input data in order to match the convolution expected data layout.
2. **Convolution operation**: You can think about this step as similar to when you effectively call the proper convolution function implementation. This step comprises possible data transformations (tiling, packing/unpacking), and microkernel function calling (BLAS, MKL).
    - **Tiling**: This step slices the convolution data into smaller chunks in order to make them fit in the system’s cache memory, for example.
    - **Packing**: Reorder the convolution data into some needed order, for example, linearizing input windows and kernel windows to simply matrix multiply them later.
    - **Microkernel**: The effective worker of the convolution algorithm, that is, the procedure that multiplies and accumulates to obtain an output element. They are usually related to BLAS, openBLAS, BLIS, or MKL GEMM methods.
    - **Unpacking**: Since the data were broken into parts (Tiling) and reorganized (Packed), in order to maintain the input data layout it should be necessary to reorder the output elements to match the input data layout. 
3. **Post-Convolution Unpacking**: return the output data to the original layout in order to match the next step's expected layout.

There is also a **Total Conv** time that accounts for the entire 2nd step; and a **Total Operation** time for all steps.

All these names are well-known concepts in the literature, however, it is easy to get their usage mixed and messy when comparing the performance of different frameworks. For example, usually, it is not clear what kind of timing the total convolution time accounts for; does it already sum up the pre-convolution packing and post-convolution unpacking time? what if the compared framework already gets the input and weight data reordered from compiling time and thus this pre/post conv packing is zero? it won’t be fair.

Due to the above considerations, we propose a Timing class that encompasses all these concepts and is inherited by ConvBench. Simply add `<time_name>_start()` and `<time_name>_update()` methods enclosed by a TIME directive in the right points of your convolution algorithm.

## ConvBench Usage

### Setting up the environment

To set up the environment, we provide a Docker file with all environment-related topics set. Follow these steps:

1. Clone the repository. 
2. Build the docker image with the following command.
``` bash
podman build --build-arg USER_ID=$(id -u ${USER}) --build-arg GROUP_ID=$(id -g ${USER}) --build-arg VSCODE_COMMIT=<vscode commit hash> -t convbench:22.04 .
```
3. Run the docker image including the repository path as follows.
``` bash
podman run --name <container_name> --rm -it --userns=keep-id -v /path/to/the/repo/ConvBench:/home/user/ConvBench/ -w /home/user/ convbench:22.04
```
4. You are ready to go!

If you prefer not to use Docker, you can build the environment by yourself.  In this case, the project dependencies are:
- Clang 14
- Python3.10
	- Pandas, 
    - matplotlib, 
    - ipywidgets, 
    - notebook, 
    - timm, 
    - torch, torchvision and torchaudio CPU version.
- pystring library (bin and include folder)
- System-wide available fmt
- System-wide available omp
- System-wide available openblas

> [!CAUTION]
> You may encounter linker problems. Try including the path of the dependency to LD_LIBRARY_PATH environment variable before running.

### Using the Benchmark

For easy usage of ConvBench, we offer an [**all-in-one jupyter notebook interface**](ConvBench.ipynb) that groups all ConvBench functionality with a configuration interface. Simply implement the convolution algorithm, select the filtering and hyperparameter options, and, finally, hit the execute button to run the experiments and automatically plot some insightful graphs.

Jupyter Notebook preview:
- Basic and Advanced filtering interface formulary:

![filtering](utils/filtering.png)
![adv filtering](utils/adv_filtering.png)

- Experimental Procedure Hyperparameter setting:

![hp details](utils/hp_details.png)

- Automatic Plotting Algorithms

![final plots](utils/plots.png)