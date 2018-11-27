#ifndef MULTIGRAPH_H
#define MULTIGRAPH_H

#include "Graph.hpp"

class Multigraph {
public:
	unsigned int num_subgraphs;
	unsigned int num_labels;
	unsigned int *idcluster;
	Graph *original;
	Graph *frontier;
	Graph **subgraphs;
	
	Multigraph(Graph *g);
	~Multigraph();
	void print();
};

#endif
