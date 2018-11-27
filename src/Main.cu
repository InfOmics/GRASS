/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Gasol v1.0 by Ming & Frank
 * 
 * Compile: make
 * Run:     ./gasol -gff -s test/match/graph_alb_N1k_K5_L10.txt test/match/q8n7e.txt
 *
 * Contacts: marco "dot" dittamondi "at" gmail "dot" com
 * 	     l "dot" gugole "at" gmail "dot" com
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

#include <iostream>
#include <ostream>
#include <pthread.h>

#include "Parser.hpp"
#include "Multigraph.hpp"
#include "Matcher.hpp"

using namespace std;

double timedelta = 0.0, totalt = 0.0;
struct timeval t_a, t_b, tta, ttb;

void usage()
{
	cout << "Usage: ./GRASS [-gffd|-gff] [-s|-c] <db filename> <query filename>" << endl;
	cout << endl << "Arguments:" << endl;
	cout << "[-gffd|-gff]:	  -gffd for direct graphs, -gff for undirect graphs" << endl;
	cout << "<serial>:	  -s for serial, -c for cuda" << endl;
	cout << "<db filename>:	  is the file of graph db" << endl;
	cout << "<query filename>: is the file of queries" << endl;
}

int main(int argc, char **argv)
{
	gettimeofday(&tta, NULL);

	//cout << "============================================" << endl;
	//cout << "===============" <<  CYEL << " GPGPU-SubGI " << CNOR << "================" << endl;
	//cout << "============================================" << endl;
	
	if (argc != 5) {
		usage();
		return 0;
	}
	
	if (strcmp(argv[1], "-gffd") != 0 && strcmp(argv[1], "-gff") != 0) {
		usage();
		return 0;
	}
	
	bool direct = (strcmp(argv[1], "-gffd") == 0) ? true : false;
	bool serial = true;
	
	if (strcmp(argv[2], "-s") != 0 && strcmp(argv[2], "-c")) {
		usage();
		return 0;
	}
	
	if (strcmp(argv[2], "-c") == 0)
		serial = false;
	
	char *graphpath = argv[3];
	char *querypath = argv[4];
	
	//
	// Graph parsing
	//
	start_timer(&t_a);

	Parser *parser = new Parser();
	GraphDB *db = parser->parseDB(graphpath, direct, false);
	GraphDB *brokenDB = db;
	//cout << "Database info" << endl;
	//brokenDB->print();

	if (! db) {
		cout << "Error during file parsing!" << endl;
		return 0;
	}	

	stop_timer(&t_a, &t_b, &timedelta, "database parsing");

	//
	// Query reading, partitioning and indexing
	//
	start_timer(&t_a);

	GraphDB *brokenQuery;
	GraphDB *query = parser->parseDB(querypath, direct, true);
	
		if (! query) {
			cout << "Query file not correct!" << endl;
			return 0;
		}
	
	brokenQuery = query;
	//cout << "Query info" << endl;
	//brokenQuery->print();

	stop_timer(&t_a, &t_b, &timedelta, "query parsing");

	//
	// Matching phase
	//
	start_timer(&t_a);

	Matcher matcher(brokenDB, brokenQuery);
	matcher.startMatching(serial);

	stop_timer(&t_a, &t_b, &timedelta, "matching");
	
	//cleaning	
	delete brokenQuery;
	delete brokenDB;
	delete parser;	

	gettimeofday(&ttb, NULL);
	totalt = (double)(ttb.tv_sec - tta.tv_sec) + (ttb.tv_usec - tta.tv_usec) / 1.0e+6;
	cout << "Total time [s]: " << totalt << endl;
	
	return 0;
}
