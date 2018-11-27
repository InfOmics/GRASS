#include "Multigraph.hpp"

Multigraph::Multigraph(Graph *g)
{
	//copy graph
	this->original = g->clone();
}

Multigraph::~Multigraph()
{
	if (this->idcluster) delete this->idcluster;
	if (this->original) delete this->original;	
	for (unsigned int i = 0; i < this->num_subgraphs; i++) if (subgraphs[i]) delete subgraphs[i];	
	if (subgraphs) delete subgraphs;
	//if (frontier) delete frontier;
}

void Multigraph::print()
{
	cout << "Multigraph info:" << endl;
	cout << " * Subgraphs: " << this->num_subgraphs << endl;
	cout << endl;
}
