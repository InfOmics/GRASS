# GRASS
An Efficient Implementation of a Subgraph Isomorphism Algorithm for GPUs.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [](#lang-en)

<hr />

The subgraph isomorphism problem is a computational task that applies to a wide range of today’s applications,
ranging from the understanding of biological networks to the
analysis of social networks. Even though different implementations for CPUs have been proposed to improve the efficiency
of such a graph search algorithm, they have shown to be
bounded by the intrinsic sequential nature of the algorithm.
More recently, graphics processing units (GPUs) have become
widespread platforms that provide massive parallelism at low
cost. Nevertheless, parallelizing any efficient and optimized
sequential algorithm for subgraph isomorphism on many-core
architectures is a very challenging task. This article presents
GRASS, a parallel implementation of the subgraph isomorphism
algorithm for GPUs. Different strategies are implemented in
GRASS to deal with the space complexity of the graph searching
algorithm, the potential workload imbalance, and the thread
divergence involved by the non-homogeneity of actual graphs.
The paper presents the results obtained on several graphs of
different sizes and characteristics to understand the efficiency of
the proposed approach.

<hr />

## Requirements

GRASS is implemented in C++/CUDA and it has been run on a Intel
Core i7-5960X 64bit hardware with 16 3GHz CPUs and 64Gb
of host RAM and running a Ubuntu 16.04 LTS operating
system, equipped with an NVIDIA GeForce GTX 980 Ti
GPU card with 6Gb of RAM and running with the CUDA
8.0 toolkit. 

<hr />

## Usage

Before running the compilation for the GRASS executable,please be sure that the CUDA toolkit has been correctly installed on your system and that the nvcc compiler is accessible form the curretn directory.

```bash
git clone https://github.com/InfOmics/GRASS.git
cd GRASS
mkdir obj
make
```

After the compilation, the GRASS executable will be available.

```bash
./GRASS -gff -c target_graph_file query_graph_file
```

where `target_graph_file` and `query_graph_file` contains two graphs in the undirected graph format (see [the RI project](https://github.com/InfOmics/RI) )


<hr />

## License
GRASS is distributed under the MIT license. This means that it is free for both academic and commercial use. 
Note however that some third party components in GRASS require that you reference certain works in scientific publications.
You are free to link or use GRASS inside source code of your own program. If do so, please reference (cite) GRASS and this website. 
We appreciate bug fixes and would be happy to collaborate for improvements. 
<!--- [License](https://raw.githubusercontent.com/GiugnoLab/PanDelos/master/LICENSE) -->

<hr />

## Citation

If you have used any of the GRASS project software, please cite the following paper:

```
Vincenzo Bonnici, Rosalba Giugno, and Nicola Bombieri. 
An Efficient Implementation of a Subgraph Isomorphism Algorithm for GPUs.
2018 IEEE International Conference on Bioinformatics and Biomedicine (BIBM). IEEE, 2018.
```
