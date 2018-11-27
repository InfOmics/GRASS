/*
 * Dynamic Bitset Class powered by Mingz & Frank
 * Version 1.0
 */

#ifndef DYNAMICBITSET_H
#define DYNAMICBITSET_H

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <cmath>

using namespace std;

#define BIT_PER_DATASET_TYPE 8

typedef unsigned char dataset_t;

class DynamicBitset {
public:
	DynamicBitset(unsigned int size);
	~DynamicBitset();
	
	void reset();
	void allset();
	void set(unsigned int pos, bool value);
	bool get(unsigned int pos);
	unsigned int onset(unsigned int length);
	
	//debug print
	void print();
	
//private:
	unsigned int size;
	unsigned int words;
	dataset_t *dataset;
};

#endif
