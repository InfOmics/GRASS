#include "GraphDB.hpp"

GraphDB::GraphDB() {}

GraphDB::~GraphDB()
{
	//delete name
	if (this->name) delete this->name;
	
	//delete graphs or multigraphs
	if (this->graphs != NULL) {
		
		for (unsigned int i = 0; i < this->num_graphs; i++) if (this->graphs[i]) delete this->graphs[i];		
		if (this->graphs) delete this->graphs;
		
	} else if (this->multigraphs != NULL) {
		
		for (unsigned int i = 0; i < this->num_graphs; i++) if (this->multigraphs[i]) delete this->multigraphs[i];		
		if (this->multigraphs) delete this->multigraphs;
	}
}

unsigned int GraphDB::getTotalGraphs()
{
	unsigned int totalsub = 0;
	
	for (unsigned int i = 0; i < this->num_graphs; i++)
		totalsub += this->multigraphs[i]->num_subgraphs;
		
	return totalsub;
}

void GraphDB::print()
{
	if (this->graphs != NULL) {
		
		cout << "General normalgraph database info:" << endl;
		cout << " * Database name: " << this->name << endl;
		cout << " * Total graphs: " << this->num_graphs << endl;
		cout << " * Total labels: " << this->num_labels << endl << endl;
	
		//for (int i = 0; i < this->num_graphs; i++)
		//	this->graphs[i]->print();
			
	} else if (this->multigraphs != NULL) {
		
		cout << "General multigraph database info:" << endl;
		cout << " * Database name: " << this->name << endl;
		cout << " * Total multigraphs: " << this->num_graphs << endl;
		cout << " * Total labels: " << this->num_labels << endl << endl;
		
		//for (int i = 0; i < this->num_graphs; i++)
		//	this->multigraphs[i]->print();
	}
}
