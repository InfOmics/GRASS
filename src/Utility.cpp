#include "Utility.hpp"

/*
 * time utility
 */
void start_timer(struct timeval *t_a)
{
	gettimeofday(t_a, NULL);
}

void stop_timer(struct timeval *t_a, struct timeval *t_b, double *timedelta, char *label)
{
	gettimeofday(t_b, NULL);
	*timedelta = (double)(t_b->tv_sec - t_a->tv_sec) + (t_b->tv_usec - t_a->tv_usec) / 1.0e+6;
	cout << "Time for " << label << " [s]:" << *timedelta << endl;
}

/*
 * display utility
 */
void print_status(unsigned int gr, unsigned int totgr, unsigned char label, unsigned int start, unsigned int end, unsigned int total, bool writing)
{	
	if (! writing) {
		if (start == 0) {
			fprintf(stderr, "%sSearching paths in graph %d/%d with label: %d (from %d to %d / %d)%s", CGRE, gr, totgr, (int)label, start, end, total, CNOR);
		} else {
			fprintf(stderr, "\r");
			fprintf(stderr, "%sSearching paths in graph %d/%d with label: %d (from %d to %d / %d)%s", CGRE, gr, totgr, (int)label, start, end, total, CNOR);
		}
	} else {
		fprintf(stderr, "\r");
		fprintf(stderr, "                                                                                     ");
		fprintf(stderr, "\r");
		fprintf(stderr, "Writing paths on disk [!]");			
	}
}

void delete_status()
{
	fprintf(stderr, "\r");
	fprintf(stderr, "                                                                                         ");
	fprintf(stderr, "\r");
}

/*
 * filename utility
 */
string filenameFromInfo(unsigned int multigraphid, unsigned int subgraphid, int level, string prefix, string ext)
{
	stringstream mgss, sgss, lss;
	mgss << multigraphid;
	sgss << subgraphid;
	lss << level;
	
	return prefix + mgss.str() + "sg" + sgss.str() + "l" + lss.str() + ext;
}

/*
 * misc utility
 */
int wcompare(int i, int j, int **weights)
{
	for (int w = 0; w < 3; w++) {
		if (weights[i][w] != weights[j][w])
			return weights[j][w] - weights[i][w];
	}
	
	return 0;
}
 
void quicksort(int *ns, int beg, int end, int **weights)
{	
	int piv, tmp;
	int l, r, p;
	
	while (beg < end) {
		l = beg; p = (beg + end) / 2; r = end;
		piv = ns[p];
		
		while (true) {
			while ((l <= r) && (wcompare(ns[l],piv,weights) <= 0 )) l++;
			while ((l <= r) && (wcompare(ns[r],piv,weights) > 0 )) r--;
			if (l > r) break;
			tmp = ns[l]; ns[l] = ns[r]; ns[r] = tmp;
			if (p == r) p = l;
			l++; r--;
		}
		
		ns[p] = ns[r]; ns[r] = piv;
		r--;
		
		if ((r - beg) < (end - l)) {
			quicksort(ns, beg, r, weights);
			beg = l;
		} else {
			quicksort(ns, l, end, weights);
			end = r;
		}
	}
}

bool searchArray(unsigned int *A, unsigned int lengthA, unsigned int *B, unsigned int lengthB, bool equal)
{
	unsigned int pA = 0, pB = 0;
	
	if (lengthA < lengthB)
		return false;
	
	if (!equal) {
		while (true) {	
			while (true) {	
				if (A[pA] >= B[pB]) {		
					pB++;
					pA++;
				
					if (pB == lengthB)
						return true;
			
				} else {			
					if (pB != 0)
						pB--;
					else
						break;
				}		
			
				if (pA == lengthA)
					return false;
			}		
		
			pA++;
		
			if (pA == lengthA)
					return false;
		}
	} else {
		for (unsigned int i = 0; i < lengthB; i++) {
			
			bool guard = false;
			for (unsigned int j = 0; j < lengthA; j++) {
				
				if (A[j] == B[i]) {
					guard = true;
					break;
				}
			}
			if (!guard)
				return false;
		}
		return true;
	}
}
