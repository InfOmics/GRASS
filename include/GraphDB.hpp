#ifndef GRAPHDB_H
#define GRAPHDB_H

#include "Graph.hpp"
#include "Multigraph.hpp"

class GraphDB {
public:
	char *name;
	unsigned int num_graphs;
	unsigned int num_labels;
	Graph **graphs;
	Multigraph **multigraphs;
	
	GraphDB();
	~GraphDB();
	unsigned int getTotalGraphs();
	void print();
};

#endif
