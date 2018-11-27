#ifndef GRAPH_H
#define GRAPH_H

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "DynamicBitset.hpp"
#include "Utility.hpp"

using namespace std;

#define VOID_NODE 0

//dfs structs
struct dfs_path {
	unsigned int *path;
	unsigned int *end_idx;
	unsigned int *current_idx;
};

struct dfs_info {
	unsigned int str_idx;
	unsigned int end_idx;
};

enum edgetype {
	EdgeType_NULL,
	EdgeType_IN, 
	EdgeType_OUT, 
	EdgeType_UN
};

struct nbrs_idx {
	unsigned int inidx;	
	unsigned int outidx;
	unsigned int inoutidx;
	unsigned int endidx;
};

class Graph {

public:	
	char *name;
	unsigned int num_labels;
	unsigned int num_nodes;
	unsigned int num_edges;
	unsigned int *adj_nodes;
	unsigned int *adj_edges;	
	unsigned int *lab_nodes;
	unsigned char *labels;
	
	unsigned int *in_nbrs;
	unsigned int *out_nbrs;
	
	struct nbrs_idx *nbrsidx;
	unsigned int nbrslength;
	unsigned int *nbrslist;
	
	string *origlabels;
	
	Graph();
	~Graph();
	
	Graph *clone();
	
	unsigned int is_in_list(unsigned int idx, unsigned int *bfs, unsigned int len_bfs);
	unsigned int neighbors(unsigned int index);
	bool hasNeighbor(unsigned int a, unsigned int b);
	bool isNeighbor(unsigned int a, unsigned int b);
	unsigned int countNeighbors(unsigned int node, enum edgetype et);
	unsigned int *uniqueEdges(unsigned int node, unsigned int *length, enum edgetype et);
	bool checkNeighborsInOut(unsigned int gnode, unsigned int qnode, Graph *q, enum edgetype et);
	bool checkNeighborsLabels(unsigned int gnode, unsigned int qnode, Graph *q, enum edgetype et);
	void print();

	//DFS strategy
	void dfs_visit(int level, unsigned int **data, int sidx, int eidx, struct dfs_info *dfsi, unsigned int num_labels);
	void dfs_memory(struct dfs_info *dfsi, unsigned int label);
};

#endif
