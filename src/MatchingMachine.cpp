#include "MatchingMachine.hpp"

MatchingMachine::MatchingMachine(Graph *g)
{
	this->graph = g;
	
	nofStates = this->graph->num_nodes;
	parentType = new int[this->nofStates];	
	parentState = new int[this->nofStates];
	patternNodes = new int[this->nofStates];
	siForPnode = new int[this->nofStates];
	parentNode = new int[this->nofStates];
		
	weights = new int*[this->nofStates];
	for (int i = 0; i < this->nofStates; i++)
		weights[i] = new int[3];
	
	nstates = new int[this->nofStates];
	
	this->build();
}

MatchingMachine::~MatchingMachine()
{
	delete parentState;
	delete patternNodes;
	delete parentType;
	delete siForPnode;
	delete parentNode;	
	for (int i = 0; i < this->nofStates; i++) delete weights[i];
	delete weights;
	delete nstates;
}

void MatchingMachine::build()
{
	int maxi = 0, maxv = 0;	
	for (int i = 0; i < nofStates; i++) {
		nstates[i] = NS_UNV;			
		weights[i][0] = 0;
		weights[i][1] = 0;
		
		weights[i][2] = graph->countNeighbors(i, EdgeType_UN);		//TODO:check why there is false flag in origina code!
			
		if (weights[i][2] > maxv) {
			maxv = weights[i][2];
			maxi = i;
		}
		
		siForPnode[i] = -1;
	}
	
	patternNodes[0] = maxi;
	parentType[maxi] = EdgeType_NULL;
	parentNode[maxi] = -1;
	
	int si = 0, ni, nni, n, tswap, nqueueL = 0, nqueueR = 1;	
	n = maxi;
	unsigned int uiolen = 0;
	unsigned int *uio = graph->uniqueEdges(n, &uiolen, EdgeType_UN);	//TODO:check why there is false flag in origina code!
	for (unsigned int i = 0; i < uiolen; i++) {		
		ni = uio[i];
		
		if (ni != n)
			weights[ni][1]++;
	}	
	delete uio;
	
	while (si < nofStates) {
		
		//if queue is empty....
		if (nqueueL == nqueueR) {
			//work on a new connected component and search for next node with highest degree (or so)				
			maxi = -1;
			maxv = -1;
			
			for (unsigned int i = 0; i < graph->num_nodes; i++) {
				n = i;				
				if (nstates[n] == NS_UNV &&  weights[n][2] > maxv){
					maxv = weights[n][2];
					maxi = n;
				}
			}
			
			patternNodes[si] = maxi;
			parentType[maxi] = EdgeType_NULL;
			parentNode[maxi] = -1;
			
			nqueueR++;
			
			n = maxi;
			
			unsigned int uiolen = 0;
			unsigned int *uio = graph->uniqueEdges(n, &uiolen, EdgeType_UN);	//TODO:check why there is false flag in origina code!
			for (unsigned int i = 0; i < uiolen; i++) {
				ni = uio[i];
				
				if (ni != n)
					weights[ni][1]++;
			}
		}
		
		//get first in queue
		n = patternNodes[si];
		siForPnode[n] = si;
		
		nqueueL++;
	
		//update nodes' flags & weights
		nstates[n] = NS_CORE;
		
		unsigned int uiolen = 0;
		unsigned int *uio = graph->uniqueEdges(n, &uiolen, EdgeType_UN);		
		for (unsigned int i = 0; i < uiolen; i++) {		
			ni = uio[i];				
				
			if (ni != n) {				
				weights[ni][0]++;
				weights[ni][1]--;	
					
				if (nstates[ni] == NS_UNV) { 
					nstates[ni] = NS_CNEIGH;
					parentNode[ni] = n;						
					//parentType[ni] = EdgeType_OUT;
					
					//add to queue
					patternNodes[nqueueR] = ni;
					nqueueR++;
						
					unsigned int uiolen = 0;
					unsigned int *uio = graph->uniqueEdges(ni, &uiolen, EdgeType_UN);						
					for (unsigned int j = 0; j < uiolen; j++) {
						nni = uio[j];
						weights[nni][1]++;
						
						if (siForPnode[ni] != -1)
							increase(patternNodes, siForPnode[nni], weights, si + 1);
					}					
					delete uio;					
				}
					
				if (siForPnode[ni] != -1)
						increase(patternNodes, siForPnode[ni], weights, si + 1);
			}
		}
		delete uio;
		
		si++;
	}
	
	bool typeO, typeI;
	for (int i = 0; i < nofStates; i++) {		
		if (parentNode[i] == -1) {			
			parentState[siForPnode[i]] = -1;
			this->parentType[siForPnode[i]] = EdgeType_NULL;
		} else {
			parentState[siForPnode[i]] = siForPnode[parentNode[i]];
				
			typeI = graph->hasNeighbor(i, parentNode[i]);
			typeO = graph->hasNeighbor(parentNode[i], i);
				
			if (typeI && typeO)
				this->parentType[siForPnode[i]] = EdgeType_UN;
			else if (typeI)
				this->parentType[siForPnode[i]] = EdgeType_IN;
			else
				this->parentType[siForPnode[i]] = EdgeType_OUT;
		}	
	}
}

int MatchingMachine::wcompare(int i, int j, int **weights)
{
	for(int w = 0;  w < 3; w++)
		if(weights[i][w] != weights[j][w])
			return weights[j][w] - weights[i][w];
	return 0;
}
	
	
	
void MatchingMachine::increase(int *ns, int i, int **weights, int leftLimit)
{
	int temp;
	while (i > leftLimit && (this->wcompare(ns[i], ns[i - 1], weights) < 0)) {
		temp = ns[i - 1];
		ns[i - 1] = ns[i];
		ns[i] = temp;
	}
}
