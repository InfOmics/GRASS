#ifndef PARSER_H
#define PARSER_H

#include <fstream>
#include <ostream>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include <vector>
#include <sstream>

#include "Graph.hpp"
#include "GraphDB.hpp"
#include "Utility.hpp"

using namespace std;

#define COMMENT_PREFIX 			'#'
#define VLS 				"|"
#define COLUMN_SEPARATOR 		" "
#define PREFIX_TREE_DB_FINAL_PATH 	"./output/db/ftmg"
#define PREFIX_TREE_QUERY_FINAL_PATH 	"./output/query/ftq"
#define PREFIX_FILE_EXT			".gpf"

struct chunk_info {
	unsigned long long int offset;
	unsigned long long int memory;
	unsigned long long int start;
	
	//disk-read phase (level > 3)
	unsigned long long int byte_size;
	unsigned long long int byte_red;
};

struct edge {
	unsigned int from;
	unsigned int to;
};

class Parser {
private:

	int level;
	unsigned int total_graphs;
	
	vector <string> *global_labels;

public:
	Parser(int level);
	Parser();
	~Parser();
	
	GraphDB *parseDB(char *file, bool direct, bool query);
	ofstream **openStreams(unsigned int multigraphid, unsigned int subgraphid, bool query);
	void save_index_array(GraphDB *db, ofstream **out, unsigned int *index, unsigned long long int length, int level, unsigned char first_label);
	void read_final(int level);
};

#endif
