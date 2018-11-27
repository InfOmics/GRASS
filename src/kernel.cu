#include "kernel.h"

#ifdef CUDA_PRINT
#include "../src/cuPrintf.cu"
#endif

/*
 * Cuda Graph allocation
 */
struct cuda_graph *cuda_graph_alloc(Graph *g, bool match)
{
	struct cuda_graph *cg = new struct cuda_graph;
	
	cudaError_t err;
	
	//build graph data structures
	unsigned int *device_adj_nodes;
	unsigned int *device_adj_edges;	
	unsigned int *device_lab_nodes;
	unsigned char *device_labels;
	err = cudaMalloc(&device_adj_nodes, sizeof(unsigned int) * g->num_nodes);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	err = cudaMemcpy(device_adj_nodes, g->adj_nodes, sizeof(unsigned int) * g->num_nodes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	err = cudaMalloc(&device_adj_edges, sizeof(unsigned int) * g->num_edges);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	err = cudaMemcpy(device_adj_edges, g->adj_edges, sizeof(unsigned int) * g->num_edges, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	err = cudaMalloc(&device_lab_nodes, sizeof(unsigned int) * g->num_nodes);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	err = cudaMemcpy(device_lab_nodes, g->lab_nodes, sizeof(unsigned int) * g->num_nodes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	err = cudaMalloc(&device_labels, sizeof(unsigned char) * g->num_nodes);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	err = cudaMemcpy(device_labels, g->labels, sizeof(unsigned char) * g->num_nodes, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }	
	
	//matching graph data structures
	unsigned int *device_in_nbrs;
	unsigned int *device_out_nbrs;
	unsigned int *device_nbrslist;
	struct nbrs_idx *device_nbrsidx;
	if (match) {
		err = cudaMalloc(&device_in_nbrs, sizeof(unsigned int) * g->num_nodes);
		if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
		err = cudaMemcpy(device_in_nbrs, g->in_nbrs, sizeof(unsigned int) * g->num_nodes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
		err = cudaMalloc(&device_out_nbrs, sizeof(unsigned int) * g->num_nodes);
		if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
		err = cudaMemcpy(device_out_nbrs, g->out_nbrs, sizeof(unsigned int) * g->num_nodes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
		err = cudaMalloc(&device_nbrslist, sizeof(unsigned int) * g->nbrslength);
		if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
		err = cudaMemcpy(device_nbrslist, g->nbrslist, sizeof(unsigned int) * g->nbrslength, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }	
		err = cudaMalloc(&device_nbrsidx, sizeof(struct nbrs_idx) * g->num_nodes);
		if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }	
		err = cudaMemcpy(device_nbrsidx, g->nbrsidx, sizeof(struct nbrs_idx) * g->num_nodes, cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	}
	
	Graph *tempgraph = new Graph();		
	tempgraph->num_nodes = g->num_nodes;
	tempgraph->num_edges = g->num_edges;
	tempgraph->num_labels = g->num_labels;
	tempgraph->nbrslength = g->nbrslength;	
	tempgraph->adj_nodes = device_adj_nodes;
	tempgraph->adj_edges = device_adj_edges;
	tempgraph->lab_nodes = device_lab_nodes;
	tempgraph->labels = device_labels;	
	
	if (match) {
		tempgraph->in_nbrs = device_in_nbrs;
		tempgraph->out_nbrs = device_out_nbrs;
		tempgraph->nbrslist = device_nbrslist;
		tempgraph->nbrsidx = device_nbrsidx;
	}
	
	Graph *device_graph = new Graph();
	err = cudaMalloc(&device_graph, sizeof(Graph));
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	err = cudaMemcpy(device_graph, tempgraph, sizeof(Graph), cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return NULL; }
	
	cg->tempgraph = tempgraph;
	cg->device_graph = device_graph;
	
	return cg;
}

/*
 * Cuda Graph free
 */
void cuda_graph_free(struct cuda_graph *cg, bool match)
{
	cudaFree(cg->tempgraph->adj_nodes);
	cudaFree(cg->tempgraph->adj_edges);	
	cudaFree(cg->tempgraph->lab_nodes);
	cudaFree(cg->tempgraph->labels);
	
	if (match) {
		cudaFree(cg->tempgraph->in_nbrs);
		cudaFree(cg->tempgraph->out_nbrs);
		cudaFree(cg->tempgraph->nbrslist);
		cudaFree(cg->tempgraph->nbrsidx);	
	}
	
	cudaFree(cg->device_graph);
	
	cg->tempgraph->name = NULL;
	cg->tempgraph->adj_nodes = NULL;
	cg->tempgraph->adj_edges = NULL;
	cg->tempgraph->lab_nodes = NULL;
	cg->tempgraph->labels = NULL;
	cg->tempgraph->in_nbrs = NULL;
	cg->tempgraph->out_nbrs = NULL;
	cg->tempgraph->nbrslist = NULL;
	cg->tempgraph->nbrsidx = NULL;
	cg->tempgraph->origlabels = NULL;
		
	delete cg->tempgraph;
	delete cg;
}

/*
 * Cuda multiarray allocation on global memory
 */
struct cuda_multiarray *cuda_multiarray_alloc(unsigned int **data, unsigned int *counters, unsigned int length)
{
	struct cuda_multiarray *cma = new cuda_multiarray;	
	unsigned int **device_multiarray;
	cudaError_t err = cudaMalloc(&device_multiarray, sizeof(unsigned int *) * length);
	if (err != cudaSuccess) { cout << "Cuda multiarray mem error!" << endl; return NULL; }
	
	cma->device_multiarray = device_multiarray;
	
	unsigned int **tempmultiarray = new unsigned int*[length];
	
	cma->tempmultiarray = tempmultiarray;
	
	for (unsigned int i = 0; i < length; i++) {
		unsigned int *device_temp;
		err = cudaMalloc(&device_temp, sizeof(unsigned int) * counters[i]);
		if (err != cudaSuccess) { cout << "Cuda multiarray mem error!" << endl; return NULL; }
		err = cudaMemcpy(device_temp, data[i], sizeof(unsigned int) * counters[i], cudaMemcpyHostToDevice);
		if (err != cudaSuccess) { cout << "Cuda multiarray mem error!" << endl; return NULL; }
		
		tempmultiarray[i] = device_temp;
	}
	
	cma->length = length;
	
	err = cudaMemcpy(device_multiarray, tempmultiarray, sizeof(unsigned int *) * length, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda multiarray mem error!" << endl; return NULL; }
	
	return cma;
}

/*
 * Cuda multiarray free on global memory
 */
void cuda_multiarray_free(struct cuda_multiarray *cma)
{
	for (unsigned int i = 0; i < cma->length; i++)
		cudaFree(cma->tempmultiarray[i]);
		
	cudaFree(cma->device_multiarray);
	
	delete cma->tempmultiarray;
	delete cma;
}

/*
 * Cuda matrix allocation on global memory
 */
struct cuda_matrix *cuda_matrix_alloc(unsigned int xdim, unsigned int ydim)
{
	struct cuda_matrix *cmm = new cuda_matrix;	
	unsigned int **device_matrix;
	cudaError_t err = cudaMalloc(&device_matrix, sizeof(unsigned int *) * ydim);
	if (err != cudaSuccess) { cout << "Cuda matrix mem error!" << endl; return NULL; }
	
	cmm->device_matrix = device_matrix;
	
	unsigned int **tempmatrix = new unsigned int*[ydim];
	
	cmm->tempmatrix = tempmatrix;
	
	for (unsigned int i = 0; i < ydim; i++) {
		unsigned int *device_temp;
		err = cudaMalloc(&device_temp, sizeof(unsigned int) * xdim);
		if (err != cudaSuccess) { cout << "Cuda matrix mem error!" << endl; return NULL; }
		
		tempmatrix[i] = device_temp;
	}
	
	err = cudaMemcpy(device_matrix, tempmatrix, sizeof(unsigned int *) * ydim, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda matrix mem error!" << endl; return NULL; }
	
	return cmm;
}

/*
 * Cuda matrix free on global memory
 */
void cuda_matrix_free(struct cuda_matrix *cmm)
{
	for (unsigned int i = 0; i < cmm->ydim; i++)
		cudaFree(cmm->tempmatrix[i]);
		
	cudaFree(cmm->device_matrix);
	
	delete cmm->tempmatrix;
	delete cmm;
}

/*
 * Print Cuda Memory usage (unreliable)
 */
void cuda_print_memusage()
{
	//show memory usage of GPU
	size_t free_byte ;
	size_t total_byte ;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte) ;
	if (cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}

	double free_db = (double)free_byte ;
	double total_db = (double)total_byte ;
	double used_db = total_db - free_db ;
        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
        	used_db / 1024.0 / 1024.0,
        	free_db / 1024.0 / 1024.0,
        	total_db / 1024.0 / 1024.0);
}

/*
 * Parallel Indexing phase functions
 */
__global__ void cuda_dfs_visit(Graph *g, int level, unsigned int **data, unsigned int sidx, unsigned int eidx, unsigned int thread_size, unsigned int grid, unsigned int total_labels)
{
	//4 max level supported
	__device__ __shared__ unsigned int path[THREAD_PER_BLOCK * 5];
	__device__ __shared__ unsigned int current_idx[THREAD_PER_BLOCK * 5];
	__device__ __shared__ unsigned int end_idx[THREAD_PER_BLOCK * 5];
	
	data[0][0] += eidx - sidx;
	
	if (blockIdx.x + grid * threadIdx.x < eidx - sidx || thread_size == 1)
	{
		int pointer = threadIdx.x * (level + 1);
		int str_thr = pointer;
		int end_thr = pointer + level;
		
		path[pointer] = g->lab_nodes[sidx + blockIdx.x + grid * threadIdx.x];
		current_idx[pointer] = g->adj_nodes[g->lab_nodes[sidx + blockIdx.x + grid * threadIdx.x]];
		end_idx[pointer] = g->out_nbrs[g->lab_nodes[sidx + blockIdx.x + grid * threadIdx.x]] + g->adj_nodes[g->lab_nodes[sidx + blockIdx.x + grid * threadIdx.x]];
		
		while (current_idx[threadIdx.x * (level + 1)] < end_idx[threadIdx.x * (level + 1)]) {			
			
			if (cuda_is_in_list(g->adj_edges[current_idx[pointer]], path, str_thr, pointer + 1) == 0) {
				
				pointer++;
				path[pointer] = g->adj_edges[current_idx[pointer - 1]];
				current_idx[pointer] = g->adj_nodes[g->adj_edges[current_idx[pointer - 1]]];
				end_idx[pointer] = g->out_nbrs[g->adj_edges[current_idx[pointer - 1]]] + g->adj_nodes[g->adj_edges[current_idx[pointer - 1]]];
				
				//create prefix tree index and save it
				unsigned long long int findex = 0;
				for (int j = 1; j < pointer - threadIdx.x * (level + 1) + 1; j++){
					findex += g->labels[path[j + threadIdx.x * (level + 1)]] * cuda_pow(total_labels, pointer - threadIdx.x * (level + 1) - j);	
				}				

				atomicAdd(&data[pointer - threadIdx.x * (level + 1)][findex], 1);
				
			} else {
				current_idx[pointer]++;
			}
			
			if (pointer == end_thr) {
				
				pointer--;
				current_idx[pointer]++;
			}
			
			while (current_idx[pointer] == end_idx[pointer] && pointer > (level + 1) * threadIdx.x) {
				pointer--;
				current_idx[pointer]++;				
			}
		}
	}
}

__device__ inline unsigned long long int cuda_pow(unsigned int a, unsigned int b)
{
	unsigned long long int ret = 1;
	
	for (unsigned int i = 0; i < b; i++)
		ret *= a;
		
	return ret;
}

__device__ inline unsigned int cuda_is_in_list(unsigned int idx, unsigned int *bfs, unsigned int start_idx, unsigned int len_bfs)
{
	unsigned int is = 0;
	
	for (int k = start_idx; k < len_bfs; k++){
		if (idx == bfs[k])
			is = 1;
	}
	
	return is;
}

/*
 * Node Match function - Checks node correctness
 */
__device__ bool cuda_nodeMatch(Graph *g, Graph *q, int tnode, int qnode, unsigned char *dynamicbitset, int *matchedNodes, int si)
{
	//check if tnode is already in matchedNodes
	//if (cuda_dynamicbitset_get(tnode, dynamicbitset))
	//	return false;
	
	//check in edges
	if (g->in_nbrs[tnode] < q->in_nbrs[qnode])
		return false;
		
	//check out edges
	if (g->out_nbrs[tnode] < q->out_nbrs[qnode])
		return false;
		
	//check labels
	if (g->labels[tnode] != q->labels[qnode])
		return false;
	
	return true;
}

/*
 * Edge Match function - Checks edge correctness
 */
__device__ bool cuda_edgeMatch(Graph *g, Graph *q, int si, int tnode, int *matchedNodes)
{
	for (int i = 0; i < si; i++)
		if(matchedNodes[i] == tnode)
			return false;

	int pneigh;
	unsigned int *uio = &q->nbrslist[q->nbrsidx[patternNodes[si]].inidx];
	unsigned int uiolen = (unsigned int) q->nbrsidx[patternNodes[si]].outidx - q->nbrsidx[patternNodes[si]].inidx;	
	for (unsigned int i = 0; i < uiolen; i++) {
		pneigh = uio[i];
		
		if (siForPnode[pneigh] <= si)
			if (! cuda_hasNeighbor(g, matchedNodes[siForPnode[pneigh]], tnode))
				return false;	
	}
		
	uio = &q->nbrslist[q->nbrsidx[patternNodes[si]].outidx];
	uiolen = (unsigned int) q->nbrsidx[patternNodes[si]].inoutidx - q->nbrsidx[patternNodes[si]].outidx;
	for (unsigned int i = 0; i < uiolen; i++) {
		pneigh = uio[i];
		
		if (siForPnode[pneigh] <= si)
			if (! cuda_hasNeighbor(g, tnode, matchedNodes[siForPnode[pneigh]]))
				return false;
	}	

	return true;
}


/*
 * Returns true if edge a -> b exists
 */
__device__ bool cuda_hasNeighbor(Graph *g, unsigned int a, unsigned int b)
{	
	for (unsigned int i = 0; i < g->out_nbrs[a]; ++i)
		if (g->adj_edges[g->adj_nodes[a] + i] == b)
			return true;
			
	return false;
}

/*
 * Returns true if edge b -> a exists
 */
__device__ bool cuda_isNeighbor(Graph *g, unsigned int a, unsigned int b)
{
	return cuda_hasNeighbor(g, b, a);
}

/*
 * Constant memory load function
 * Must be in the same file where arrays are decleared __const__ o.0
 */
void cuda_load_constant_mem(MatchingMachine *mama)
{
	cudaError_t err = cudaMemcpyToSymbol(parentType, mama->parentType, sizeof(int) * mama->nofStates, 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda constant memory error!" << endl; return; }
	
	err = cudaMemcpyToSymbol(parentState, mama->parentState, sizeof(int) * mama->nofStates, 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda constant memory error!" << endl; return; }
	
	err = cudaMemcpyToSymbol(patternNodes, mama->patternNodes, sizeof(int) * mama->nofStates, 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda constant memory error!" << endl; return; }
	
	err = cudaMemcpyToSymbol(siForPnode, mama->siForPnode, sizeof(int) * mama->nofStates, 0, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda constant memory error!" << endl; return; }
}

/*
 * Decode function - It identifies the unique root-node path
 */
__device__ void cuda_decode(unsigned int p, unsigned char length, unsigned short *A, unsigned int *ar, unsigned char str)
{
	unsigned int temp = p;
	for (int j = length - 1; j >= str; j--) {
		A[j] = temp % ar[j];
		temp = temp / ar[j];
	}
}

/*
 * Flooding step function
 */
__global__ void cuda_match(Graph *g, Graph *q, unsigned int length, unsigned int *lsolutions, unsigned int **candidates,
                           unsigned char *bitset, unsigned int *querynodes, int *m, int workthreads, int step)
{
	unsigned int thread_id = (blockIdx.x * MFST + threadIdx.x) + step;

	if (thread_id < workthreads) {

		int matchedNodes[QN];
		unsigned short localpath[QN];

		cuda_decode(thread_id, length, localpath, lsolutions, 0);	  

		for (unsigned int i = 0; i < length; i++) {
			matchedNodes[i] = candidates[i][localpath[i]];
		}

		bool guard = true;
		for (unsigned int j = 0; j < length; j++) {
			if (! cuda_edgeMatch(g, q, j, matchedNodes[j], matchedNodes)) {
				guard = false;
				break;
			}
		}

		if (guard) {
			bitset[thread_id] = 1;
			atomicAdd(m, 1);

		} else {
			bitset[thread_id] = 0;
		}	
	}
}

/*
 * Hypersearch visit function
 */
__global__ void cuda_blockmatch(Graph *g, Graph *q, unsigned int length, unsigned int *_lsolutions, unsigned int **candidates,
                                unsigned char *bitset, int *m, int workthreads, int *_cidx, int *_eidx, int *_l_v,
                                unsigned int l_v_size, int start, int qn)
{
	//cuPrintf("Block: %d, Thread %d\n",blockIdx.x,threadIdx.x);
	if (threadIdx.x < workthreads) {
		extern __shared__ int shared[];
		
		int ptr = 0;
		__shared__ int cur[1];		
		
		int *cidx = &shared[0];			
		int eidx[QN];
		int l_v[QN];
		unsigned int lsolutions[QN];

		unsigned char *lbit = (unsigned char*) &shared[qn];
		unsigned short matchedNodesOffset[QN];
		int matchedNodes[QN];

		int blc = blockIdx.x + 1 + start;
		if (threadIdx.x == 0) {
			int c = 1;
			for (int i = 0; i < MFST * MFSG * MSTEP; i++) { 			
				if (bitset[i] == 1) { 
					if (c == blc) {
						*cur = i;
						break;
					} else {
						c++;
					}
				}
			}

			for (int i = 0; i < l_v_size; i++)
				cidx[i] = _cidx[i];
		}

		for (int i = 0; i < l_v_size; i++) {
			eidx[i] = _eidx[i];
			l_v[i] = _l_v[i];
		}
		for (int i = 0; i < QN; i++)
			lsolutions[i] = _lsolutions[i];

		__syncthreads();

		cuda_decode(*cur, l_v[ptr], &matchedNodesOffset[threadIdx.x * qn * 0], lsolutions, 0);

		ptr++;

		if (threadIdx.x < eidx[ptr]) {
			cuda_decode(threadIdx.x, l_v[ptr], &matchedNodesOffset[threadIdx.x * qn * 0], lsolutions, l_v[ptr - 1]);

			for (unsigned int j = 0; j < l_v[ptr]; j++) {
				matchedNodes[j] = candidates[j][matchedNodesOffset[threadIdx.x * qn * 0 + j]];
			}

			bool guard = true;			
			for (unsigned int j = l_v[ptr - 1]; j < l_v[ptr]; j++) {

				if (! cuda_edgeMatch(g, q, j, matchedNodes[j], matchedNodes)) {
					guard = false;
					break;
				}
			}			

			if (guard) {
				lbit[ptr * workthreads + threadIdx.x] = 1;
				if (ptr == l_v_size - 1) atomicAdd(m, 1);		

			} else {
				lbit[ptr * workthreads + threadIdx.x] = 0;
			}
		}

		__syncthreads();	

		if (ptr < l_v_size - 1) {

			if (threadIdx.x == 0) {
				for (int j = 0; j < eidx[ptr]; j++) {
					cidx[ptr] = workthreads;

					if (lbit[ptr * workthreads + j] == 1) {
						cidx[ptr] = j;
						lbit[ptr * workthreads + j] = 0; 
						break;
					}				
				}
			}

			__syncthreads();

			while (ptr > 0) {	

				if (ptr < l_v_size - 1) {
					cuda_decode(cidx[ptr], l_v[ptr], &matchedNodesOffset[threadIdx.x * qn * 0], lsolutions, l_v[ptr - 1]);	
				}

				if (cidx[ptr] < eidx[ptr] && ptr < l_v_size - 1) {

					ptr++;

					if (threadIdx.x < eidx[ptr]) {

						cuda_decode(threadIdx.x, l_v[ptr], &matchedNodesOffset[threadIdx.x * qn * 0], lsolutions, l_v[ptr - 1]);

						for (unsigned int j = 0; j < l_v[ptr]; j++) {
							matchedNodes[j] = candidates[j][matchedNodesOffset[threadIdx.x * qn * 0 + j]];
						}

						bool guard = true;					
						for (unsigned int j = l_v[ptr - 1]; j < l_v[ptr]; j++) {
							if (! cuda_edgeMatch(g, q, j, matchedNodes[j], matchedNodes)) {
								guard = false;
								break;
							}
						}

						if (guard) {
							lbit[ptr * workthreads + threadIdx.x] = 1;

							if (ptr == l_v_size - 1)
								atomicAdd(m, 1);

						} else {
							lbit[ptr * workthreads + threadIdx.x] = 0;
						}
					} 

					if (ptr < l_v_size - 1) {
						__syncthreads();

						if (threadIdx.x == 0) {
							cidx[ptr] = workthreads;
							for (int j = 0; j < eidx[ptr]; j++) {
								if (lbit[ptr * workthreads + j] == 1) {
									cidx[ptr] = j;
									lbit[ptr * workthreads + j] = 0;
									break;
								}
							}
						}
					}
				} else {

					ptr--;

					if (ptr > 0 && threadIdx.x == 0) {

						cidx[ptr] = workthreads;
						for (int j = 0; j < eidx[ptr]; j++) {
							if (lbit[ptr * workthreads + j] == 1) {
								cidx[ptr] = j;
								lbit[ptr * workthreads + j] = 0;
								break;
							}
						}
					}
				}

				__syncthreads();
			}

		}
	}
}











//
