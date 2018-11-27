#include "Graph.hpp"

Graph::Graph() {}

Graph::~Graph()
{
	if (this->name) delete this->name;
	if (this->adj_nodes) delete this->adj_nodes;
	if (this->adj_edges) delete this->adj_edges;	
	if (this->lab_nodes) delete this->lab_nodes;	
	if (this->labels) delete this->labels;
	if (this->in_nbrs) delete this->in_nbrs;
	if (this->out_nbrs) delete this->out_nbrs;
	if (this->nbrsidx) delete this->nbrsidx;
	if (this->nbrslist) delete this->nbrslist;
	if (this->origlabels) delete[] this->origlabels;
}

unsigned int Graph::is_in_list(unsigned int idx, unsigned int *bfs, unsigned int len_bfs)
{
	unsigned int is = 0;
	
	for (int k = 0; k < len_bfs; k++){
		if (idx == bfs[k])
			is = 1;
	}
	
	return is;
}

/*
 * return number of neighbors (out edges)
 */
unsigned int Graph::neighbors(unsigned int index)
{	
	unsigned int nbr = 0;
	unsigned int start = this->adj_nodes[index];
	unsigned int end_idx = index + 1;
	
	if (start != VOID_NODE) {
		
		if (index < this->num_nodes - 1) {
			
			while (this->adj_nodes[end_idx] == VOID_NODE && end_idx <  this->num_nodes)
				end_idx++;
			
			if (end_idx <  this->num_nodes) {
				nbr = this->adj_nodes[end_idx] - this->adj_nodes[index];
			} else {
				nbr = this->num_edges - this->adj_nodes[index];
			}
		} else {
			nbr = this->num_edges - this->adj_nodes[index];
		}
	}
	
	return nbr;
}

/*
 * return the list of unique number of neighbors of type et
 */
unsigned int *Graph::uniqueEdges(unsigned int node, unsigned int *length, enum edgetype et)
{
	bool intype = false, outtype = false;
	if (et == EdgeType_IN) {
		intype = true;
	} else if (et == EdgeType_OUT) {
		outtype = true;
	} else if (et == EdgeType_UN) {
		intype = true;
		outtype = true;
	}
	
	vector <unsigned int> temp;
	
	//in edges
	if (intype) {
		for (unsigned int i = 0; i < this->num_nodes; i++)
			if (this->isNeighbor(node, i))
				temp.push_back(i);
	}
	
	//out edges
	if (outtype) {
		for (unsigned int i = 0; i < this->neighbors(node); i++) {
			unsigned int tn = this->adj_edges[this->adj_nodes[node] + i];
		
			bool guard = true;
			if (intype) {
				for (unsigned int j = 0; j < temp.size(); j++)
					if (temp.at(j) == tn)
						guard = false;
			}
				
			if (guard)
				temp.push_back(tn);
		}
	}
	
	*length = temp.size();
	unsigned int *nodes = new unsigned int[temp.size()];
	memcpy(nodes, &temp[0], temp.size() * sizeof(unsigned int));
	
	return nodes;
}

/*
 * return the unique number of neighbors of type et
 */
unsigned int Graph::countNeighbors(unsigned int node, enum edgetype et)
{
	bool intype = false, outtype = false;
	if (et == EdgeType_IN) {
		intype = true;
	} else if (et == EdgeType_OUT) {
		outtype = true;
	} else if (et == EdgeType_UN) {
		intype = true;
		outtype = true;
	}
	
	unsigned int tc = 0;
	DynamicBitset *already = new DynamicBitset(this->num_nodes);
	
	//in edges
	if (intype) {
		for (unsigned int i = 1; i < this->num_edges; i++) {
			if (this->adj_edges[i] == node) {
				already->set(this->adj_edges[i], true);
				tc++;
			}
		}
	}
	
	//out edges
	if (outtype) {
		for (unsigned int i = 0; i < this->neighbors(node); i++) {
			unsigned int tn = this->adj_edges[this->adj_nodes[node] + i];
			
			if (! already->get(tn))
				tc++;
		}
	}	
	delete already;
	
	return tc;
}

bool mypredicate(unsigned int i, unsigned int j) {
	return (i >= j);
}

bool Graph::checkNeighborsInOut(unsigned int gnode, unsigned int qnode, Graph *q, enum edgetype et)
{
	bool state = true;
	
	unsigned int *ginnbrs = new unsigned int[this->in_nbrs[gnode]];
	unsigned int *goutnbrs = new unsigned int[this->out_nbrs[gnode]];
	unsigned int *qinnbrs = new unsigned int[q->in_nbrs[qnode]];
	unsigned int *qoutnbrs = new unsigned int[q->out_nbrs[qnode]];
	
	for (unsigned int i = 0; i < this->in_nbrs[gnode]; i++) ginnbrs[i] = this->in_nbrs[this->nbrslist[this->nbrsidx[gnode].inidx + i]];
	for (unsigned int i = 0; i < this->out_nbrs[gnode]; i++) goutnbrs[i] = this->out_nbrs[this->nbrslist[this->nbrsidx[gnode].outidx + i]];
	for (unsigned int i = 0; i < q->in_nbrs[qnode]; i++) qinnbrs[i] = q->in_nbrs[q->nbrslist[q->nbrsidx[qnode].inidx + i]];
	for (unsigned int i = 0; i < q->out_nbrs[qnode]; i++) qoutnbrs[i] = q->out_nbrs[q->nbrslist[q->nbrsidx[qnode].outidx + i]];
	
	sort(ginnbrs, ginnbrs + this->in_nbrs[gnode]);
	sort(goutnbrs, goutnbrs + this->out_nbrs[gnode]);
	sort(qinnbrs, qinnbrs + q->in_nbrs[qnode]);
	sort(qoutnbrs, qoutnbrs + q->out_nbrs[qnode]);
	
	/*if (gnode == 1 && qnode == 2){
	cout << "gin: " << endl;
	for (unsigned int i = 0; i < this->in_nbrs[gnode]; i++) cout << ginnbrs[i] << endl;
	cout << "qin: " << endl;
	for (unsigned int i = 0; i < q->in_nbrs[qnode]; i++) cout << qinnbrs[i] << endl;
	}*/
	
	//using frank comparison
	if (q->in_nbrs[qnode] != 0) {
		if (! searchArray(ginnbrs, this->in_nbrs[gnode], qinnbrs, q->in_nbrs[qnode], false)) {
			state = false;
		} else {
			if (q->out_nbrs[qnode] != 0) {
				if (! searchArray(goutnbrs, this->out_nbrs[gnode], qoutnbrs, q->out_nbrs[qnode], false))
					state = false;
			}
		}
	}
	
	//if (gnode == 1 && qnode == 2)
	//	cout << "@@ " << state << endl;
	
	delete ginnbrs;
	delete goutnbrs;
	delete qinnbrs;
	delete qoutnbrs;
	
	return state;
}

bool Graph::checkNeighborsLabels(unsigned int gnode, unsigned int qnode, Graph *q, enum edgetype et)
{	
	//versione frank
	bool state = true;
	
	unsigned int *ginlabels = new unsigned int[this->in_nbrs[gnode]];
	unsigned int *goutlabels = new unsigned int[this->out_nbrs[gnode]];
	unsigned int *qinlabels = new unsigned int[q->in_nbrs[qnode]];
	unsigned int *qoutlabels = new unsigned int[q->out_nbrs[qnode]];
	
	for (unsigned int i = 0; i < this->in_nbrs[gnode]; i++) ginlabels[i] = this->labels[this->nbrslist[this->nbrsidx[gnode].inidx + i]];
	for (unsigned int i = 0; i < this->out_nbrs[gnode]; i++) goutlabels[i] = this->labels[this->nbrslist[this->nbrsidx[gnode].outidx + i]];
	for (unsigned int i = 0; i < q->in_nbrs[qnode]; i++) qinlabels[i] = q->labels[q->nbrslist[q->nbrsidx[qnode].inidx + i]];
	for (unsigned int i = 0; i < q->out_nbrs[qnode]; i++) qoutlabels[i] = q->labels[q->nbrslist[q->nbrsidx[qnode].outidx + i]];
	
	/*
	sort(ginlabels, ginlabels + this->in_nbrs[gnode]);
	sort(goutlabels, goutlabels + this->out_nbrs[gnode]);
	sort(qinlabels, qinlabels + q->in_nbrs[qnode]);
	sort(qoutlabels, qoutlabels + q->out_nbrs[qnode]);
	*/	
	
	/*
	//using frank comparison
	if (q->in_nbrs[qnode] != 0) {
		if (! searchArray(ginlabels, this->in_nbrs[gnode], qinlabels, q->in_nbrs[qnode], true)) {
			state = false;
		} else {
			if (q->out_nbrs[qnode] != 0) {
				if (! searchArray(goutlabels, this->out_nbrs[gnode], qoutlabels, q->out_nbrs[qnode], true))
					state = false;
			}
		}
	}*/
	
	//printf("gnode: %d qnode: %d\n", gnode, qnode);
	bool again[this->in_nbrs[gnode]];
	if (q->in_nbrs[qnode] != 0) {	
		for (unsigned int i = 0; i < q->in_nbrs[qnode]; i++) {		
			bool guard = false;
			
			memset(again, 0, sizeof(bool) * this->in_nbrs[gnode]);
			for (unsigned int j = 0; j < this->in_nbrs[gnode]; j++) {			
			
				if (qinlabels[i] == ginlabels[j] && !again[j]) {
					guard = true;
					again[j] = true;
					break;
				}			
			}
		
			if (! guard) {
				state = false;
				break;
			}
		}		
		
		//cout << "int state: " << state << endl;
		
		bool again_2[this->out_nbrs[gnode]];
		for (unsigned int i = 0; i < q->out_nbrs[qnode]; i++) {
			bool guard = false;
			
			memset(again_2, 0, sizeof(bool) * this->out_nbrs[gnode]);
			for (unsigned int j = 0; j < this->out_nbrs[gnode]; j++) {
				if (qoutlabels[i] == goutlabels[j] && !again_2[j]) {
					guard = true;
					again_2[j] = true;
					break;
				}			
			}
		
			if (! guard) {
				state = false;
				break;
			}
		}	
	}
	
	//if (state) cout << "true!" << endl;
	//else cout << "false!" << endl;
	
	delete ginlabels;
	delete goutlabels;
	delete qinlabels;
	delete qoutlabels;
	
	return state;
}

/*
 * DFS strategy 
 */
void Graph::dfs_memory(struct dfs_info *dfsi, unsigned int label)
{
	dfsi->end_idx = dfsi->str_idx;
	
	if (dfsi->str_idx >= this->num_nodes) {
		return;
	}
	
	bool isPresent = false;
	for (unsigned int i = 0; i < this->num_nodes; i++) {
		if (label == this->labels[i]) {
			isPresent = true;
			break;	
		}
	}
	
	if (! isPresent)
		return;	
	
	while (this->labels[this->lab_nodes[dfsi->end_idx]] == this->labels[this->lab_nodes[dfsi->str_idx]] && dfsi->str_idx < this->num_nodes) {
		dfsi->end_idx++;
		
		if (dfsi->end_idx == this->num_nodes) break;
	}
}

void Graph::dfs_visit(int level, unsigned int **data, int sidx, int eidx, struct dfs_info *dfsi, unsigned int num_labels)
{
	struct dfs_path dfsp;
	dfsp.path = new unsigned int[level + 1];
	dfsp.end_idx = new unsigned int[level + 1];
	dfsp.current_idx = new unsigned int[level + 1];
	
	data[0][0] += eidx - sidx;
	
	for (unsigned int i = sidx; i < eidx; i++) {
				
		int pointer = 0;
		dfsp.path[pointer] = this->lab_nodes[i];
		dfsp.current_idx[pointer] = this->adj_nodes[this->lab_nodes[i]];
		dfsp.end_idx[pointer] = this->neighbors(this->lab_nodes[i]) + this->adj_nodes[this->lab_nodes[i]];
		
		while (dfsp.current_idx[0] < dfsp.end_idx[0]) {			
			
			if (is_in_list(this->adj_edges[dfsp.current_idx[pointer]], dfsp.path, pointer + 1) == 0) {
				
				pointer++;
				dfsp.path[pointer] = this->adj_edges[dfsp.current_idx[pointer - 1]];
				dfsp.current_idx[pointer] = this->adj_nodes[this->adj_edges[dfsp.current_idx[pointer - 1]]];
				dfsp.end_idx[pointer] = this->neighbors(this->adj_edges[dfsp.current_idx[pointer - 1]]) + this->adj_nodes[this->adj_edges[dfsp.current_idx[pointer - 1]]];
				
				unsigned long long int findex = 0;
				for (int j = 1; j < pointer + 1; j++)
					findex += this->labels[dfsp.path[j]] * pow(num_labels, pointer - j);
				
				data[pointer][findex]++;
				
			} else {
				dfsp.current_idx[pointer]++;
			}
			
			if (pointer == level) {		
				pointer--;
				dfsp.current_idx[pointer]++;
			}
			
			while (dfsp.current_idx[pointer] == dfsp.end_idx[pointer] && pointer > 0) {
				pointer--;
				dfsp.current_idx[pointer]++;				
			}
		}
	}
	
	//cleaning
	delete dfsp.path;
	delete dfsp.current_idx;
	delete dfsp.end_idx;
}

//a -> b
bool Graph::hasNeighbor(unsigned int a, unsigned int b)
{	
	for (unsigned int i = 0; i < this->neighbors(a); i++)
		if (this->adj_edges[this->adj_nodes[a] + i] == b)
			return true;
			
	return false;
}

//a <- b
bool Graph::isNeighbor(unsigned int a, unsigned int b)
{
	return this->hasNeighbor(b, a);
}

//TODO: make this in copy contructor
Graph *Graph::clone()
{
	Graph *g = new Graph();
	
	//copy of name
	string name (this->name);
	g->name = new char[name.size() + 1];
	g->name[name.size()] = '\0';
	memcpy(g->name, this->name, name.size());
	
	//copy of nums
	g->num_labels = this->num_labels;
	g->num_nodes = this->num_nodes;
	g->num_edges = this->num_edges;
	
	//copy of adj lists
	g->adj_nodes = new unsigned int[this->num_nodes];
	memcpy(g->adj_nodes, this->adj_nodes, this->num_nodes * sizeof(unsigned int));
	g->adj_edges = new unsigned int[this->num_edges];
	memcpy(g->adj_edges, this->adj_edges, this->num_edges * sizeof(unsigned int));
	g->lab_nodes = new unsigned int[this->num_nodes];
	memcpy(g->lab_nodes, this->lab_nodes, this->num_nodes * sizeof(unsigned int));
	g->labels = new unsigned char[this->num_nodes];
	memcpy(g->labels, this->labels, this->num_nodes * sizeof(unsigned char));
	
	//copy in/out neighbors
	g->in_nbrs = new unsigned int[this->num_nodes];
	memcpy(g->in_nbrs, this->in_nbrs, this->num_nodes * sizeof(unsigned int));
	g->out_nbrs = new unsigned int[this->num_nodes];
	memcpy(g->out_nbrs, this->out_nbrs, this->num_nodes * sizeof(unsigned int));
	
	//copy neighbors index and list
	g->nbrsidx = new struct nbrs_idx[this->num_nodes];
	for (unsigned int i = 0; i < this->num_nodes; i++) {
		g->nbrsidx[i].inidx = this->nbrsidx[i].inidx;
		g->nbrsidx[i].outidx = this->nbrsidx[i].outidx;
		g->nbrsidx[i].endidx = this->nbrsidx[i].endidx;
	}		
	g->nbrslist = new unsigned int[this->nbrslength];
	memcpy(g->nbrslist, this->nbrslist, this->nbrslength * sizeof(unsigned int));
	
	g->origlabels = new string[this->num_nodes];
	for (unsigned int i = 0; i < this->num_nodes; i++)
		g->origlabels[i] = this->origlabels[i];
	
	return g;
}

void Graph::print()
{
	cout << "Graph name: " << this->name << endl;
	cout << " * Nodes: " << this->num_nodes << endl;
	cout << " * Edges: " << this->num_edges << endl;
	cout << " * Labels: " << this->num_labels << endl;
	
	cout << " * Node list: ";
	for (unsigned int i = 0; i < this->num_nodes; i++)
		cout << this->adj_nodes[i] << ((i != this->num_nodes) ? " " : "");
	cout << endl;
	
	cout << " * Edges list: ";
	for (unsigned int i = 0; i < this->num_edges; i++)
		cout << this->adj_edges[i] << ((i != this->num_edges) ? " " : "");
	cout << endl;
	
	cout << " * Label list: ";
	for (unsigned int i = 0; i < this->num_nodes; i++)
		cout << (int)this->labels[i] << ((i != this->num_nodes) ? " " : "");
	cout << endl;
	
	cout << " * Labnodes list: ";
	for (unsigned int i = 0; i < this->num_nodes; i++)
		cout << this->lab_nodes[i] << ((i != this->num_nodes) ? " " : "");
	cout << endl;
	
	cout << " * In neighbors counters: ";
	for (unsigned int i = 0; i < this->num_nodes; i++)
		cout << this->in_nbrs[i] << ((i != this->num_nodes) ? " " : "");
	cout << endl;
	
	cout << " * Out neighbors counters: ";
	for (unsigned int i = 0; i < this->num_nodes; i++)
		cout << this->out_nbrs[i] << ((i != this->num_nodes) ? " " : "");
	cout << endl;
	
	cout << " * Neighbors list: ";
	for (unsigned int i = 0; i < this->nbrslength; i++)
		cout << this->nbrslist[i] << ((i != this->nbrslength) ? " " : "");		
	cout << endl;
	
	cout << endl;
}














/*
	//versione frank
	bool state = true;
	
	unsigned int *ginlabels = new unsigned int[this->in_nbrs[gnode]];
	unsigned int *goutlabels = new unsigned int[this->out_nbrs[gnode]];
	unsigned int *qinlabels = new unsigned int[q->in_nbrs[qnode]];
	unsigned int *qoutlabels = new unsigned int[q->out_nbrs[qnode]];
	
	for (unsigned int i = 0; i < this->in_nbrs[gnode]; i++) ginlabels[i] = this->labels[this->nbrslist[this->nbrsidx[gnode].inidx + i]];
	for (unsigned int i = 0; i < this->out_nbrs[gnode]; i++) goutlabels[i] = this->labels[this->nbrslist[this->nbrsidx[gnode].outidx + i]];
	for (unsigned int i = 0; i < q->in_nbrs[qnode]; i++) qinlabels[i] = this->labels[q->nbrslist[q->nbrsidx[qnode].inidx + i]];
	for (unsigned int i = 0; i < q->out_nbrs[qnode]; i++) qoutlabels[i] = this->labels[q->nbrslist[q->nbrsidx[qnode].outidx + i]];
	
	sort(ginlabels, ginlabels + this->in_nbrs[gnode]);
	sort(goutlabels, goutlabels + this->out_nbrs[gnode]);
	sort(qinlabels, qinlabels + q->in_nbrs[qnode]);
	sort(qoutlabels, qoutlabels + q->out_nbrs[qnode]);

	/*
	//using stl comparison
	unsigned int *y;
	y = search(&ginlabels[0], &ginlabels[this->in_nbrs[gnode]], &qinlabels[0], &qinlabels[this->in_nbrs[qnode]]);
	if (*y != this->in_nbrs[gnode] - 1)
		state = false;
	else {
		y = search(&goutlabels[0], &goutlabels[this->out_nbrs[gnode]], &qoutlabels[0], &qoutlabels[this->out_nbrs[qnode] - 1]);
		if (*y != this->out_nbrs[gnode] - 1)
			state = false;
	}
	
	
	//using frank comparison
	if (q->in_nbrs[qnode] != 0) {
		if (! searchArray(ginlabels, this->in_nbrs[gnode], qinlabels, q->in_nbrs[qnode], true)) {
			state = false;
		} else {
			if (q->out_nbrs[qnode] != 0) {
				if (! searchArray(goutlabels, this->out_nbrs[gnode], qoutlabels, q->out_nbrs[qnode], true))
					state = false;
			}
		}
	}
	
	delete ginlabels;
	delete goutlabels;
	delete qinlabels;
	delete qoutlabels;
	
	return state;













	bool state = true;

	vector<unsigned int> ginlabels; vector<unsigned int>::iterator itgin;
	vector<unsigned int> goutlabels; vector<unsigned int>::iterator itgout;
	vector<unsigned int> qinlabels; vector<unsigned int>::iterator itqin;
	vector<unsigned int> qoutlabels; vector<unsigned int>::iterator itqout;
	
	for (unsigned int i = 0; i < this->in_nbrs[gnode]; i++)	ginlabels.push_back(this->nbrslist[this->nbrsidx[gnode].inidx + i]);
	for (unsigned int i = 0; i < this->out_nbrs[gnode]; i++) goutlabels.push_back(this->nbrslist[this->nbrsidx[gnode].outidx + i]);
	for (unsigned int i = 0; i < q->in_nbrs[qnode]; i++) qinlabels.push_back(q->nbrslist[q->nbrsidx[qnode].inidx + i]);
	for (unsigned int i = 0; i < q->out_nbrs[qnode]; i++) qoutlabels.push_back(q->nbrslist[q->nbrsidx[qnode].outidx + i]);
	
	sort(ginlabels.begin(), ginlabels.begin() + this->in_nbrs[gnode]);
	sort(goutlabels.begin(), goutlabels.begin() + this->out_nbrs[gnode]);
	sort(qinlabels.begin(), qinlabels.begin() + q->in_nbrs[qnode]);
	sort(qoutlabels.begin(), qoutlabels.begin() + q->out_nbrs[qnode]);
	
	vector<unsigned int>::iterator itin;
	vector<unsigned int>::iterator itout;
	
	if (ginlabels.size() > 0) {
		itin = search(ginlabels.begin(), ginlabels.end(), qinlabels.begin(), qinlabels.end());
		if (itin != ginlabels.end()) {		
			itout = search(goutlabels.begin(), goutlabels.end(), qoutlabels.begin(), qoutlabels.end());
		
			if (itout == goutlabels.end()) {
				state = false;
			}
		} else {
			cout << "false" << endl;
			state = false;
		}
	}
*/
