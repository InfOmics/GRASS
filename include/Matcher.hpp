#ifndef MATCHER_H
#define MATCHER_H

#include <iostream>
#include <vector>
#include <fstream>
//#include <boost/thread.hpp> 

#include "GraphDB.hpp"
#include "DynamicBitset.hpp"
#include "Utility.hpp"
#include "MatchingMachine.hpp"
#include "kernel.h"

#ifdef CUDA_PRINT
#include "cuPrintf.cuh"
#endif

#define CPU_THREADS 16

using namespace std;

struct solutions_space_s {
	unsigned int **candidates;
	unsigned int *lsolutions;
	unsigned int length;
};

class Matcher {
public:
	Matcher(GraphDB *db, GraphDB *query);
	~Matcher();

	void startMatching(bool serial);
	
	//RI functions
	bool nodeMatch(Graph *g, Graph *q, int tnode, int qnode, DynamicBitset *already);
	bool edgeMatch(Graph *g, Graph *q, int si, int tnode, int *patternNodes, int *siForPnode, int *matchedNodes);
	
	//serial match
	void match(MatchingMachine *mama, Graph *g, Graph *q);
	
	//cuda match
	void cudaMatch(MatchingMachine *mama, Graph *g, Graph *q);
	void decode(unsigned int p, unsigned char length, unsigned short *A, unsigned int *ar, unsigned char str);
	unsigned int levelBFS(unsigned int th_num, unsigned int *candidates_num, unsigned int length, unsigned int str_lv);
	void threadAssign(unsigned int *candidates, unsigned int q_len, unsigned int str_lv, unsigned int end_lv, unsigned int num_th, unsigned short **th_paths, unsigned short *path);
	unsigned int ppruning(Graph *g, Graph *q, vector<unsigned int> *va, vector<unsigned int> *vb, MatchingMachine *mama);
	struct solutions_space_s *coreCandidates(MatchingMachine *mama, Graph *g, Graph *q);
	
	//cpu parallel match
	void cpuMatch(MatchingMachine *mama, Graph *g, Graph *q, int proc);
	void *matchThread(MatchingMachine *mama, Graph *g, Graph *q, unsigned int start_node, unsigned int *match_counter);
	
private:
	GraphDB *db;
	GraphDB *query;
    double timedelta, totalt;
    struct timeval t_a, t_b, tta, ttb;
};

#endif
