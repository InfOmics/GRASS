#ifndef UTILITY_H
#define UTILITY_H

#include <iostream>
#include <sstream>
#include <ctime>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include <stdio.h>

using namespace std;

#define CLGR "\033[1;30m"
#define CYEL "\033[1;33m"
#define CRED "\033[1;31m"
#define CBLU "\033[1;34m"
#define CNOR "\033[0m"
#define CGRE "\033[0;32m"

/*
 * time utility
 */
void start_timer(struct timeval *t_a);
void stop_timer(struct timeval *t_a, struct timeval *t_b, double *timedelta, char *label);

/*
 * display utility
 */
void print_status(unsigned int gr, unsigned int totgr, unsigned char label, unsigned int start, unsigned int end, unsigned int total, bool writing);
void delete_status();

/*
 * filename utility
 */
string filenameFromInfo(unsigned int multigraphid, unsigned int subgraphid, int level, string prefix, string ext);

/*
 * misc utility
 */
int wcompare(int i, int j, int **weights);
void quicksort(int *ns, int beg, int end, int **weights);
bool searchArray(unsigned int *A, unsigned int lengthA, unsigned int *B, unsigned int lengthB, bool equal);

#endif
