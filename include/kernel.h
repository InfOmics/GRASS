#ifndef KERNEL_H
#define KERNEL_H

#include <cuda.h>
#include <math.h>

#include "Graph.hpp"
#include "MatchingMachine.hpp"

#define THREAD_PER_BLOCK 	32
#define CUDA_SMP 		80
#define QN 			40
#define GRID_X			225
#define GRID_Y			32

//match phase definitions
#define MATCH_THREAD_PER_BLOCK 	128
#define MATCH_BLOCK_PER_GRID	240
#define NODES_PER_THREAD	1
#define MAX_THREADS		1024*1024

#define MFST			1024
#define MFSG			1024
#define MSTEP			100
#define NEWMATCH_THREADS	MATCH_THREAD_PER_BLOCK * MATCH_BLOCK_PER_GRID

#define asd cuPrintf("asd\n");

/*
 * misc functions
 */
struct cuda_graph {
	Graph *tempgraph;
	Graph *device_graph;
};

struct cuda_multiarray {
	unsigned int **device_multiarray;	
	unsigned int **tempmultiarray;
	
	unsigned int length;
};

struct cuda_matrix {
	unsigned int **device_matrix;	
	unsigned int **tempmatrix;
	
	unsigned int xdim;
	unsigned int ydim;
};
 
struct cuda_graph *cuda_graph_alloc(Graph *g, bool match);
void cuda_graph_free(struct cuda_graph *cg, bool match);
struct cuda_multiarray *cuda_multiarray_alloc(unsigned int **data, unsigned int *counters, unsigned int length);
void cuda_multiarray_free(struct cuda_multiarray *cma);

struct cuda_matrix *cuda_matrix_alloc(unsigned int xdim, unsigned int ydim);
void cuda_matrix_free(struct cuda_matrix *cmm);

template<typename T> inline void cuda_memset(T *x, T value, unsigned long count)
{
	cudaError_t err;
	T *temp = new T[count / sizeof(T)];
	
	for (unsigned long i = 0; i < count / sizeof(T); i++)
		temp[i] = value;
	
	err = cudaMemcpy(x, temp, count, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda template memset error!" << endl; return; }
	
	delete temp;
}

void cuda_print_memusage();

/*
 * generic graph functions
 */
__device__ bool cuda_hasNeighbor(Graph *g, unsigned int a, unsigned int b);
__device__ bool cuda_isNeighbor(Graph *g, unsigned int a, unsigned int b);
__device__ unsigned int cuda_neighbors(Graph *g, unsigned int index);

/*
 * build phase functions
 */
__global__ void cuda_dfs_visit(Graph *g, int level, unsigned int **data, unsigned int sidx, unsigned int eidx, unsigned int thread_size, unsigned int grid, unsigned int total_labels);
__device__ inline unsigned int cuda_is_in_list(unsigned int idx, unsigned int *bfs, unsigned int start_idx, unsigned int len_bfs);
__device__ inline unsigned long long int cuda_pow(unsigned int a, unsigned int b);

/*
 * match phase functions
 */ 
__constant__ int parentType[QN];
__constant__ int parentState[QN];
__constant__ int patternNodes[QN];
__constant__ int siForPnode[QN];

void cuda_load_constant_mem(MatchingMachine *mama);

__device__ bool cuda_nodeMatch(Graph *g, Graph *q, int tnode, int qnode, unsigned char *dynamicbitset, int *matchedNodes, int si);
__device__ bool cuda_edgeMatch(Graph *g, Graph *q, int si, int tnode, int *matchedNodes);
__device__ void cuda_decode(unsigned int p, unsigned char length, unsigned short *A, unsigned int *ar, unsigned char str);
__global__ void cuda_match(Graph *g, Graph *q, unsigned int length, unsigned int *lsolutions, unsigned int **candidates,
                           unsigned char *bitset, unsigned int *querynodes, int *m, int workthreads, int step);
__global__ void cuda_blockmatch(Graph *g, Graph *q, unsigned int length, unsigned int *lsolutions, unsigned int **candidates,
				unsigned char *bitset, int *m, int workthreads, int *cidx, int *eidx, int *l_v, unsigned int l_v_size,
				int start, int qn);
#endif
