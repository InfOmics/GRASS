#include "Parser.hpp"

Parser::Parser(int level) : level(level)
{
	global_labels = new vector<string>;
}

Parser::Parser()
{
	global_labels = new vector<string>;
}

Parser::~Parser()
{
	delete global_labels;
}

GraphDB *Parser::parseDB(char *file, bool direct, bool query)
{	
	ifstream in;
	in.open(file, ios::in);
	
	if (! in.is_open())
		return NULL;
		
	GraphDB *db = new GraphDB();
	db->num_graphs = 0;
		
	vector<Graph *> data_v;		
	string temp;
	
	while (!in.eof() && getline(in, temp)) {
		
		Graph *g = new Graph();			
			
		//set name
		char *local_name = new char[temp.size() + 1];
		local_name[temp.size()] = '\0';			
		memcpy(local_name, temp.c_str(), temp.size() * sizeof(char));			
		g->name = local_name;
		
		//set labels
		getline(in, temp);			
		g->num_nodes = atoi(temp.c_str());
		g->origlabels = new string[g->num_nodes];
		
		g->labels = new unsigned char[g->num_nodes];
		
		vector <string> *v_labels = new vector<string>;
		g->num_labels = 0;	
		bool add = true, add_global = true;
		for (unsigned int i = 0; i < g->num_nodes; i++) {
			getline(in, temp);
			g->origlabels[i] = temp;
				
			for (unsigned int j = 0; j < v_labels->size(); j++) {
					
				if (temp == v_labels->at(j))
					add = false;				
			}
				
			if (add) {
				v_labels->push_back(temp);
			}
				
			add = true;
			
			for (unsigned int j = 0; j < global_labels->size(); j++) {
					
				if (temp == global_labels->at(j)) {
					g->labels[i] = j;
					add_global = false;
				}					
			}
				
			if (add_global) {
				if (query) {
					cout << "Query error! Label not found in db list! Returning NULL db!" << endl;
					return NULL;
				} else {
					g->labels[i] = global_labels->size();				
					global_labels->push_back(temp);
				}
			}
				
			add_global = true;
		}
		
		g->num_labels = v_labels->size();		
		
		//order nodes by labels
		g->lab_nodes = new unsigned int[g->num_nodes];
		vector <vector <unsigned int> > v_lab_nodes(global_labels->size());

		for (unsigned int i = 0; i < g->num_nodes; i++) {
			v_lab_nodes[g->labels[i]].push_back(i);
		}
		unsigned int oltc = 0;
		for (unsigned int i = 0; i < v_lab_nodes.size(); i++) {
			for (unsigned int j = 0; j < v_lab_nodes[i].size(); j++) {
				g->lab_nodes[oltc] = v_lab_nodes[i].at(j);
				oltc++;	
			}
		}
		
		delete v_labels;
		//set nodes and edges
		getline(in, temp);			
		g->num_edges = atoi(temp.c_str());
			
		struct edge *e = new struct edge[g->num_edges];
		
		//set in/out neighbors
		g->in_nbrs = new unsigned int[g->num_nodes];
		memset(g->in_nbrs, 0, g->num_nodes * sizeof(unsigned int));
		g->out_nbrs = new unsigned int[g->num_nodes];
		memset(g->out_nbrs, 0, g->num_nodes * sizeof(unsigned int));
		
		//set neighbors lists
		g->nbrsidx = new struct nbrs_idx[g->num_nodes];		
		vector<unsigned int>*innbrs_v = new vector<unsigned int>[g->num_nodes];
		vector<unsigned int>*inoutnbrs_v = new vector<unsigned int>[g->num_nodes];
		vector<unsigned int>*outnbrs_v = new vector<unsigned int>[g->num_nodes];
		vector<unsigned int> nbrlist_v;
		//cout << "error start" << endl;	
		//edge load: complexity = O(e)
		for (unsigned int i = 0; i < g->num_edges; i++) {
			getline(in, temp);
				
			char *ct = strtok((char *)temp.c_str(), COLUMN_SEPARATOR);
			e[i].from = atoi(ct);
				
			ct = strtok(NULL, COLUMN_SEPARATOR); //ct?
			e[i].to = atoi(ct);
			if (direct) {
				//neighbors counters
				g->in_nbrs[e[i].to]++;
				g->out_nbrs[e[i].from]++;				
				
				//neighbors lists
				innbrs_v[e[i].to].push_back(e[i].from);
				outnbrs_v[e[i].from].push_back(e[i].to);
			}
		}
		//cout << "error end" << endl;
		vector <vector <unsigned int> > v_tempnodes(g->num_nodes + 1);
		for (unsigned int i = 0; i < g->num_edges; i++)
			v_tempnodes[e[i].from].push_back(e[i].to);		
		
		if (! direct) {		
			//make graph undirect
			for (unsigned int i = 0; i < v_tempnodes.size(); i++) {
			
				for (unsigned int j = 0; j < v_tempnodes[i].size(); j++) {
				
					unsigned int temp = v_tempnodes[i].at(j);
					bool guard = true;
					for (unsigned int k = 0; k < v_tempnodes[temp].size(); k++) {
						if (v_tempnodes[temp].at(k) == i) {
							guard = false;
							break;
						}
					}
				
					if (guard) v_tempnodes[temp].push_back(i);
				}
			}
		
			g->num_edges = 1;
			for (unsigned int i = 0; i < v_tempnodes.size(); i++)
				g->num_edges += v_tempnodes[i].size();
			
		} else {
			g->num_edges++;
		}
		
		if (! direct) {
			for (unsigned int i = 0; i < v_tempnodes.size(); i++) {
				for (unsigned int j = 0; j < v_tempnodes[i].size(); j++) {
					//neighbors counters
					g->in_nbrs[v_tempnodes[i].at(j)]++;
					g->out_nbrs[i]++;
			
					//neighbors lists
					innbrs_v[v_tempnodes[i].at(j)].push_back(i);
					outnbrs_v[i].push_back(v_tempnodes[i].at(j));
				}
			}
		}
		
		unsigned int nbrstc = 0;
		for (unsigned int i = 0; i < g->num_nodes; i++) {
			g->nbrsidx[i].inidx = nbrstc;	
			g->nbrsidx[i].outidx = nbrstc + innbrs_v[i].size();			
			g->nbrsidx[i].inoutidx = nbrstc + innbrs_v[i].size() + outnbrs_v[i].size();
			
			//in copy
			for (unsigned int j = 0; j < innbrs_v[i].size(); j++)
				nbrlist_v.push_back(innbrs_v[i].at(j));
				
			//out copy
			for (unsigned int j = 0; j < outnbrs_v[i].size(); j++)
				nbrlist_v.push_back(outnbrs_v[i].at(j));
			
			//inout merge
			unsigned int iinbrstc = inoutnbrs_v[i].size();
			bool guard = false;
			for (unsigned int j = 0; j < innbrs_v[i].size(); j++) {
				for (unsigned int k = 0; k < outnbrs_v[i].size(); k++)
					if (outnbrs_v[i].at(k) == innbrs_v[i].at(j))
						guard = true;
						
				if (guard) {
					iinbrstc++;
					inoutnbrs_v[i].push_back(innbrs_v[i].at(j));
				}					
			}
			
			for (unsigned int j = 0; j < inoutnbrs_v[i].size(); j++)
				nbrlist_v.push_back(inoutnbrs_v[i].at(j));
			
			g->nbrsidx[i].endidx = nbrstc + innbrs_v[i].size() + outnbrs_v[i].size() + iinbrstc;
			
			nbrstc += innbrs_v[i].size() + outnbrs_v[i].size() + iinbrstc;
		}
		
		g->nbrslength = nbrlist_v.size();
		g->nbrslist = new unsigned int[g->nbrslength];
		memcpy(g->nbrslist, &nbrlist_v[0], sizeof(unsigned int) * g->nbrslength);
		
		delete []innbrs_v;
		delete []outnbrs_v;
		delete []inoutnbrs_v;			
		
		g->adj_nodes = new unsigned int[g->num_nodes + 1];
		g->adj_edges = new unsigned int[g->num_edges];
			
		unsigned int tc;
		if (v_tempnodes[0].size() > 0) {
			g->adj_nodes[0] = 1;
			tc = v_tempnodes[0].size();
		} else {
			g->adj_nodes[0] = VOID_NODE;
			tc = 0;
		}
			
		g->adj_edges[0] = 0;
		for (unsigned int j = 0; j < v_tempnodes[0].size(); j++)
			g->adj_edges[j + 1] = v_tempnodes[0][j];
		
		for (unsigned int i = 1; i < g->num_nodes + 1; i++) {
			if (v_tempnodes[i].size() > 0) {
				g->adj_nodes[i] = tc + 1;
				tc += v_tempnodes[i].size();
			} else {
				g->adj_nodes[i] = VOID_NODE;
			}
				
			for (unsigned int j = 0; j < v_tempnodes[i].size(); j++)
				g->adj_edges[tc - v_tempnodes[i].size() + j + 1] = v_tempnodes[i][j];
		}		
			
		data_v.push_back(g);
			
		delete e;	
		
		db->num_graphs++;
	}
	
	//set db name
	string name = query ? "Query" : "Database";
	db->name = new char[name.size() + 1];
	db->name[name.size()] = '\0';
	memcpy(db->name, name.c_str(), (name.size() + 1) * sizeof(char));
	
	//set db graphs
	Graph **data = new Graph*[data_v.size()];
	memcpy(data, &data_v[0], sizeof(Graph *) * data_v.size());
		
	db->graphs = data;
	db->multigraphs = NULL;
	db->num_labels = global_labels->size();
		
	return db;
}

ofstream **Parser::openStreams(unsigned int multigraphid, unsigned int subgraphid, bool query)
{
	ofstream **out = new ofstream*[level + 1];
	
	for (int a = 0; a < this->level + 1; a++) {
		out[a] = new ofstream;
		out[a]->open(filenameFromInfo(multigraphid, subgraphid, a, ((query) ? PREFIX_TREE_QUERY_FINAL_PATH : PREFIX_TREE_DB_FINAL_PATH), PREFIX_FILE_EXT).c_str());
	}
	
	return out;
}

void Parser::save_index_array(GraphDB *db, ofstream **out, unsigned int *index, unsigned long long int length, int level, unsigned char first_label)
{
	unsigned char arrout[4];
	
	for (unsigned long long int i = 0; i < length; i++) {
		
		if (index[i] != 0) {
			
			unsigned long long int temp = i;
			unsigned char row[level + 1];
			
			row[0] = first_label;
			
			for (int j = level; j > 0; j--) {
				row[level - j + 1] = (unsigned char)(temp / pow(db->num_labels, j - 1));
				temp -= ((unsigned long long int)row[level - j + 1]) * pow(db->num_labels, j - 1);
			}		
			
			out[level]->write((const char *)row, level + 1);			
			
			arrout[0] = index[i] & 0x000000ff;
			arrout[1] = (index[i]  & 0x0000ff00) >> 8;
			arrout[2] = (index[i]  & 0x00ff0000) >> 16;
			arrout[3] = (index[i]  & 0xff000000) >> 24;
			
			out[level]->write((const char *)arrout, 4);
		}
	}
}	

void Parser::read_final(int level)
{
	stringstream ss;
	ss << level;
	string temp_path = PREFIX_TREE_DB_FINAL_PATH + ss.str() + PREFIX_FILE_EXT;	
	ifstream in("./output/final_tree_0_level_2.gpf", ios::in | ios::binary);	
	
	unsigned long long int size_byte = 0;
	in.seekg(0, ios::end);
	size_byte = in.tellg();
	in.seekg(0, ios::beg);
	
	unsigned char inarr[level + 1];
	
	cout << endl << "Binary output [path - counter]:" << endl;
		
	unsigned long long int byte_red = 0;
	while (byte_red < size_byte) {
	
		in.read((char *)inarr, level + 1);
		
		for (int k = 0; k < level + 1; k++)			
			cout << (unsigned int)inarr[k] << ((k != level) ? VLS : "");
		
		cout << " ";	
		
		unsigned char inarr2[4];
		inarr2[0] = 0x00;
		inarr2[1] = 0x00;
		inarr2[2] = 0x00;
		inarr2[3] = 0x00;
		in.read((char *)inarr2, 4);
		
		unsigned int gc = 0;
		gc = inarr2[0];
		gc = gc | (inarr2[1] << 8);
		gc = gc | (inarr2[2] << 16);
		gc = gc | (inarr2[3] << 24);
		
		cout << "\t" << gc << endl;

		byte_red += (level + 1) + 4;
	}
	
	in.close();
}
