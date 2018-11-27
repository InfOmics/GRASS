#ifndef MATCHINGMACHINE_H
#define MATCHINGMACHINE_H

#include "Graph.hpp"
#include "Utility.hpp"

using namespace std;

class MatchingMachine {
public:
	MatchingMachine(Graph *g);
	~MatchingMachine();

	Graph *graph;
	int nofStates;	
	int *parentType;
	int *parentState;
	int *patternNodes;
	int *siForPnode;
	int *parentNode;	
	int **weights;
	int *nstates;
	
private:
	void build();
	int wcompare(int i, int j, int **weights);
	void increase(int *ns, int i, int **weights, int leftLimit);
	
	enum NodeState {
		NS_CORE,
		NS_CNEIGH,
		NS_UNV
	};
};

#endif
