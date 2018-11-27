#include "Matcher.hpp"
#include "CudaErrorCheck.cu"

Matcher::Matcher(GraphDB *db, GraphDB *query) : db(db), query(query) {}

Matcher::~Matcher() {}

/*
 * Main function of Matching phase: for each query it creates the MatchingMachine and it performs matching
 * based on serial/parallel strategy and filtering infos in candidates bitset
 */
void Matcher::startMatching(bool serial)
{
	for (unsigned int i = 0; i < this->query->num_graphs; i++) {		
		
		for (unsigned int j = 0; j < this->db->num_graphs; j++) {
									
					//declare local graph and query pointers
					Graph *g = this->db->graphs[i];
					Graph *q = this->query->graphs[i];
	
					//create matching machine for match
					MatchingMachine mama(q);

					//perform match step
					if (serial) {
						this->match(&mama, g, q);
					} else {
						this->cudaMatch(&mama, g, q);
					}
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Parallel Match
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/*
 * Decode function - It identifies the unique root-node path
 */
void Matcher::decode(unsigned int p, unsigned char length, unsigned short *A, unsigned int *ar, unsigned char str)
{
	unsigned int temp = p;
	for (int j = length - 1; j >= str; j--) {
		A[j] = temp % ar[j];
		temp = temp / ar[j];
	}
}

/*
 * It checks how many levels can be covered in one flooding step (starting from root)
	candidate_num - # candidati per nodo
	length - # livelli query
 */
unsigned int Matcher::levelBFS(unsigned int th_num, unsigned int *candidates_num, unsigned int length, unsigned int str_lv) {
	int counter = 1;
	unsigned int pointer = str_lv;
	
	if (th_num == 0) return 0;

	if (pointer == length - 1)
		return pointer + 1;
	
	//Limitiamo il numero di realthreads a 500.000.
	while (th_num >= counter && pointer <= length & counter < 500000) {
		//std:://cout"Candidates for level "<< pointer <<" : "<<candidates_num[pointer]<<"\n";
		counter *= candidates_num[pointer];
		pointer++; 
	}
	
	return pointer-1;
}

/*
 * It checks how many levels can be covered in one flooding step (starting from leaves)
 
unsigned int levelBFS2(unsigned int th_num, unsigned int *candidates_num, unsigned int length, unsigned int str_lv) {
	int counter = 1;
	unsigned int pointer = str_lv + 1;
	
	if (th_num == 0) return 0;

	
	while (th_num >= counter && pointer <= length) {
		counter *= candidates_num[pointer - 1];
		pointer++; 
	}
	
	return pointer-1;
}

/*
 * It creates the partial solution of a specific thread
 *
void Matcher::threadAssign(unsigned int *candidates, unsigned int q_len,  unsigned int str_lv, unsigned int end_lv, unsigned int num_th, unsigned short **th_paths, unsigned short *path) {
	
	for (int i = 0; i < num_th; i++) {
		decode(i, end_lv, th_paths[i], candidates, str_lv);
		
		for (char j = 0; j < str_lv; j++)
			th_paths[i][j] = path[j];
	}
}

/*
 * Cuda Match Launcher
 */
void Matcher::cudaMatch(MatchingMachine *mama, Graph *g, Graph *q)
{

	//PREPROCESSING
    timedelta = 0.0;
    bool shMeError = false;
    totalt = 0.0;
    	start_timer(&t_a);

	/*ssol è una struct con i risultati del preprocessing:
		ssol->candidates - id dei candidati;
		ssol-> lsolutions - # di candidati per nodo;
		ssol->length - # livelli query; 	
	*/
	struct solutions_space_s *ssol = this->coreCandidates(mama, g, q);	

	stop_timer(&t_a, &t_b, &timedelta, "Preprocessing");
	
	/*
	//CONTROLLO GRAFO OTTENUTO
	cout<<"#Livelli: "<<ssol->length<<endl;
	for(int i=0; i < 3; i++){
		cout<<"Livello "<<i<<"\t#Candidati per nodo: "<<ssol->lsolutions[i]<<endl;
		for(j=0; j< ssol->lsolutions[j];j++)
			cout<<"Id candidati: "<<ssol->candidates[i][j]<<endl;
	*/

	//if is NULL there is no solution and nothing to free
	if (ssol == NULL)
		return;

	//
	//cuda allocations
	//	
	struct cuda_graph *cg = cuda_graph_alloc(g, true);
	struct cuda_graph *cq = cuda_graph_alloc(q, true);	
	
	//candidates counters copy
	unsigned int *device_lsolutions;
	cudaError_t err = cudaMalloc(&device_lsolutions, sizeof(unsigned int) * ssol->length);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return; }
	err = cudaMemcpy(device_lsolutions, ssol->lsolutions, sizeof(unsigned int) * ssol->length, cudaMemcpyHostToDevice);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return; }
	
	//candidates copy
	struct cuda_multiarray *cma = cuda_multiarray_alloc(ssol->candidates, ssol->lsolutions, ssol->length);

	//match data structures		
	cuda_load_constant_mem(mama);
	
	//dim3 block(MATCH_THREAD_PER_BLOCK);
	//dim3 grid(MATCH_BLOCK_PER_GRID);	
	
	unsigned char *device_bitset;
	err = cudaMalloc(&device_bitset, sizeof(unsigned char) * MFST * MFSG * MSTEP);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return; }
	err = cudaMemset(device_bitset, 0, sizeof(unsigned char) * MFST * MFSG * MSTEP);
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return; }
	
	//device match counter
	int *device_m;	
	err = cudaMalloc(&device_m, sizeof(int));
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return; }
	cuda_memset(device_m, 0, sizeof(int));
	if (err != cudaSuccess) { cout << "Cuda memory error!" << endl; return; }

	// Visita BFS per calcolare il numero di realthreads e il livello di flooding
	vector<unsigned int> *levels_v = new vector<unsigned int>;	
	
	//starting level = 0
	unsigned int strlv = 0;
	levels_v->push_back(strlv);
	strlv = levelBFS(MFST * MFSG * MSTEP, ssol->lsolutions, ssol->length, strlv);
	//strlv è il numero di livelli copribili dal Flooding step. 
	//cout << "I can cover with Flooding step:"<< strlv <<endl;
	
	levels_v->push_back(strlv);
	
	//levels_v at 0 contiene lo starting level=0
	//levels_v at 1 contiene il numero di livelli coperti;
	
	//Ricalcolo il numero di thread che copre tutti i nodi sino al flooding level calcolato.
	int realthreads = 1;
	
	for (int i = 0; i < strlv; i++){
		realthreads *= ssol->lsolutions[i];
		//cout<<"Candidates for level "<< i <<" : "<<ssol->lsolutions[i]<<"\n";
	}
	//cout<<"Realthreads:"<<realthreads<<endl;
#ifdef CUDA_PRINT
	cudaPrintfInit();
#endif

	//Specifica la grid e la dimensione dei blocchi --> kernel.h: MFST 1024, MFSG 1024.
	dim3 dimBlock(MFST,1,1);
	dim3 dimGrid(MFSG,1);

	start_timer(&t_a);
	//Kernel launch. Ogni lancio del kernel copre MFSG*MFST realthreads. 
	//Lanci finchè non sono state coperte tutte le realthreads.
	/*NEW 13/01/2014. Stamattina abbiamo visto che poichè MFSG*MFST=1024*1024=1048576
	e realthreads è limitato da 500.000 allora ci sarà un solo lancio del Kernel. */
	for (int i = 0; i < realthreads; i += (MFSG * MFST)) {
		cuda_match<<<dimGrid, dimBlock>>>(cg->device_graph, cq->device_graph, levels_v->at(1), device_lsolutions, cma->device_multiarray, device_bitset, NULL, device_m, realthreads, i);
	}

#ifdef CUDA_PRINT       
        cudaPrintfDisplay(stdout, false);
        cudaPrintfEnd();
#endif

	int match_counter;
	err = cudaMemcpy(&match_counter, (int *)device_m, sizeof(int), cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) { 
		cout << "Cuda memory error memcpy match_counter!" << endl;
		printf("Error Code: %d\n", err);
		printf("CUDA Error Code: %s\n", cudaGetErrorString(cudaGetLastError()));
		return; 
	}        

	stop_timer(&t_a, &t_b, &timedelta, "Flooding step function");
	//cout << endl << "Starting Nodes:" << match_counter << endl;
	
	//Se il flooding level è diverso dal numero di nodi della query,
	//cioè se non sono arrivato alle foglie con il flooding...Inizia la fase di Hypersearch.
	if(strlv!=ssol->length){
		//salvo il valore del livello di flooding in strlv
		strlv = levels_v->at(1);
		//cancello e ricreo il vettore levels_v
	
		delete(levels_v);
	
		levels_v = new vector<unsigned int>;

		levels_v->push_back(strlv);
		//levels_v at 0 = flooding level+1 = sarà il nuovo starting level per la fase di hypersearch.		
		int temp_str = 0;
		//cout"MTPB:"<<MATCH_THREAD_PER_BLOCK<< endl;		
		/* Visite BFS dal primo livello sotto quello di flooding fino alle foglie. 
		 * in questo modo trova tutti i livelli completi copribili ad ogni step della visita BFS e il rispettivo numero di realthreads.  */
		while (strlv < ssol->length) {
			temp_str = strlv;			
			strlv = levelBFS(MATCH_THREAD_PER_BLOCK, ssol->lsolutions, ssol->length, strlv);
			
			if (temp_str == strlv) strlv++; //Incremento strlv se la BFS ha coperto solo il livello corrente.
			//Inserisco in coda il livello raggiunto.
			levels_v->push_back(strlv);
			
		}
		//levels_v -> size() è uguale al numero di step da eseguire (uno per ogni levelBFS lanciata)
		// per la fase di Hypersearch.
		
		//Dichiarazione array utili per la fase di hypersearch.
		int *cidx = new int[levels_v->size()];
		int *eidx = new int[levels_v->size()];
		eidx[0] = 0;
		for (int i = 0; i < levels_v->size() - 1; i++) {
			eidx[i + 1] = 1;
			cidx[i] = 0;
			//per ogni livello raggiunto in ciascuno delle visite BFS
			//memorizzo in eidx il numero di nodi.
			for (int j = levels_v->at(i); j < levels_v->at(i + 1); j++) {
				eidx[i + 1] *= ssol->lsolutions[j];
				//printf("lsolutions[%d] = %d\n", j, ssol->lsolutions[j]);	
			}
		}
		
		//a questo punto calcolo il numero di nodi massimo in uno step 
		// e il totale dei nodi.
		unsigned int totalsize = 0;
		unsigned int maxsize = 0;
		////cout"Livelli: "<<levels_v->size()<<endl;
		for (unsigned int i = 0; i < levels_v->size(); i++){
			totalsize +=eidx[i];
			//cout<<"livello "<<levels_v->at(i)<<": "<<eidx[i]<<endl;
			if (eidx[i] > maxsize){
				maxsize = eidx[i];
					
			}		
		}
		
		//allocazione variabili per il device.
	
		int *device_cidx;
        	err = cudaMalloc(&device_cidx, sizeof(int) * levels_v->size());
        	if (err != cudaSuccess) { cout << "Cuda memory error alloc cidx!" << endl; return; }
        	err = cudaMemcpy(device_cidx, cidx, sizeof(int) * levels_v->size(), cudaMemcpyHostToDevice);
        	if (err != cudaSuccess) { cout << "Cuda memory error memcpy cidx!" << endl; return; }
        
        	int *device_eidx;
        	err = cudaMalloc(&device_eidx, sizeof(int) * levels_v->size());
        	if (err != cudaSuccess) { cout << "Cuda memory error alloc eidx!" << endl; return; }
        	err = cudaMemcpy(device_eidx, eidx, sizeof(int) * levels_v->size(), cudaMemcpyHostToDevice);
        	if (err != cudaSuccess) { cout << "Cuda memory error memcpy eidx!" << endl; return; }
        

		//Copio il vettore levels_v in un array che quindi memorizzo sul device.
        	int *device_l_v;
        	int *l_v = new int[levels_v->size()];
        	for (int i = 0; i < levels_v->size(); i++)
                	l_v[i] = levels_v->at(i);

				
        	err = cudaMalloc(&device_l_v, sizeof(int) * levels_v->size());
        	if (err != cudaSuccess) { cout << "Cuda memory error alloc l_v!" << endl; return; }
        	err = cudaMemcpy(device_l_v, l_v, sizeof(int) * levels_v->size(), cudaMemcpyHostToDevice);
        	if (err != cudaSuccess) { cout << "Cuda memory error memcpy l_v!" << endl; return; }
        	delete l_v; //cancello l_v che sull'host non mi serve avendo già levels_v...
		err = cudaMemset(device_m, 0,  sizeof(int)); //resetto il match counter del device
		if (err != cudaSuccess) { cout << "Cuda memset error!" << endl; return; }
		
		//Calcolo della gridsize.
		int grid_size = (MAX_THREADS / maxsize);
		
		//lunghezza della query.
		int qn = ssol->length;
	
		//calcolo della shared_memory necessaria.
		unsigned int shared_memory = qn * sizeof(short) + (totalsize) * sizeof(int);
		cout << "shared_memory:" << shared_memory <<endl; // "\t" << "qn:" << qn << "\t"<<endl;
		//cout << "totalsize:" << totalsize << endl;
		//Max shared Memory
		cudaDeviceProp props;
		cudaGetDeviceProperties( &props, 0);
		////cout"Shared MemoryPerBlock Device:"<< props.sharedMemPerBlock<<endl;
		if(shared_memory < props.sharedMemPerBlock){

			#ifdef CUDA_PRINT
        			cudaPrintfInit();
			#endif	
	
			start_timer(&t_a);
			/*Kernel launch. Il numero di blocchi è pari al match counter cioè agli starting nodes,
				se sono in quantità minore della gridsize, altrimenti si settano $gridsize blocchi e si
				lancia il kernel più volte. */
			for (int i = 0; i < match_counter; i += grid_size) {
				int temp= i + grid_size < match_counter ? grid_size : match_counter - i;
				cout << "Number of Block:" << temp << endl;
				cout << "Number of thread:" << maxsize << endl;
				
				//HYPERSEARCH PHASE
				cuda_blockmatch<<<temp, maxsize, shared_memory>>>(cg->device_graph, cq->device_graph, ssol->length, device_lsolutions, cma->device_multiarray,device_bitset, device_m, maxsize, device_cidx, device_eidx, device_l_v, levels_v->size(), i, qn);
				CudaCheckError();

			}
			#ifdef CUDA_PRINT       
        			cudaPrintfDisplay(stdout, false);
        			cudaPrintfEnd();
			#endif
			err = cudaMemcpy(&match_counter, (int *)device_m, sizeof(int), cudaMemcpyDeviceToHost);
			if (err != cudaSuccess) { cout << "Cuda memory error memcpy match_counter!" << endl; return; }
			stop_timer(&t_a, &t_b, &timedelta, "Hypersearch visit function");
		}else{
			cout << CRED <<"Your GPU doesn't have enough shared memory to compute this query" << endl;
			cout << CRED <<"Request: " << shared_memory / 1024 << " KB" << endl;
			cout << CRED <<"Available: " << props.sharedMemPerBlock / 1024 << " KB" << endl << endl;
			shMeError=true;
		}
	}
	if(shMeError){
		cout << CGRE << "Total parallel match: " << "unknown" << endl << endl;
	}else{
		cout << CGRE << "Total parallel match: " << match_counter << endl << endl;
	}		
}

/*
 * Preprocessing function - It creates the solution space to visit
 */
struct solutions_space_s *Matcher::coreCandidates(MatchingMachine *mama, Graph *g, Graph *q)
{
	struct solutions_space_s *ssol = new struct solutions_space_s;
	ssol->lsolutions = new unsigned int[mama->nofStates];
	
	bool allpruned = false;
	
	//vector for candidates
	vector<unsigned int> candidates_v[mama->nofStates];
	
	//calcolo core_index (?) per l'euristica ppruning
	unsigned int core_index[mama->nofStates];
	for (unsigned int i = 0; i < mama->nofStates; i++) {
		unsigned int temp = 0;
		for (unsigned int j = 0; j < mama->nofStates; j++) {
			if (mama->patternNodes[j] == i) {
				temp = j;
				break;
			}
		}
		
		core_index[i] = temp;
	}	
	//check conditions
	for (unsigned int i = 0; i < mama->nofStates; i++) {
	
		for (unsigned int j = 0; j < g->num_nodes; j++) {			
			//check label
			if (g->labels[j] != q->labels[mama->patternNodes[i]]) {
				continue;	
			}
			
			//check in/out degree		
			if (g->in_nbrs[j] < q->in_nbrs[mama->patternNodes[i]] || g->out_nbrs[j] < q->out_nbrs[mama->patternNodes[i]]) {
				continue;	
			}			
			
			//if (mama->patternNodes[i] == 2) cout << mama->patternNodes[i] << " - " << j << endl;
			
			//check in/out neighbors
			if (! g->checkNeighborsInOut(j, mama->patternNodes[i], q, EdgeType_NULL)) {				
				continue;
			}			
			
			//check in/out neighbors labels
			if (! g->checkNeighborsLabels(j, mama->patternNodes[i], q, EdgeType_NULL)) {
				continue;
			}			
			
			//check level II in/out neighbors
			bool mustcont = false;
			for (unsigned int h = 0; h < q->nbrsidx[mama->patternNodes[i]].outidx - q->nbrsidx[mama->patternNodes[i]].inidx; h++) {				
				bool guard = false;
				for (unsigned int k = 0; k < g->nbrsidx[j].outidx - g->nbrsidx[j].inidx; k++) {					
					if (g->checkNeighborsInOut(g->nbrslist[g->nbrsidx[j].inidx + k], q->nbrslist[q->nbrsidx[mama->patternNodes[i]].inidx + h], q, EdgeType_NULL)) {			
						guard = true;
						break;
					}				
				}
				
				if (! guard) {
					mustcont = true;
					break;
				}
			}
			
			if (mustcont)
				continue;
			
			mustcont = false;
			for (unsigned int h = 0; h < q->neighbors(mama->patternNodes[i]); h++) {				
				bool guard = false;
				for (unsigned int k = 0; k < g->neighbors(j); k++) {
					if (g->checkNeighborsInOut(g->adj_edges[g->adj_nodes[j] + k], q->adj_edges[q->adj_nodes[mama->patternNodes[i]] + h], q, EdgeType_NULL)) {
						guard = true;
						break;
					}				
				}
				
				if (! guard) {
					mustcont = true;
					break;
				}
			}
			
			if (mustcont)
				continue;
			
			//check level II in/out neighbors labels
			mustcont = false;
			for (unsigned int h = 0; h < q->nbrsidx[mama->patternNodes[i]].outidx - q->nbrsidx[mama->patternNodes[i]].inidx; h++) {
				bool guard = false;
				for (unsigned int k = 0; k < g->nbrsidx[j].outidx - g->nbrsidx[j].inidx; k++) {					
					if (g->checkNeighborsLabels(g->nbrslist[g->nbrsidx[j].inidx + k], q->nbrslist[q->nbrsidx[mama->patternNodes[i]].inidx + h], q, EdgeType_NULL)) {
						guard = true;
						break;
					}				
				}
				
				if (! guard) {
					mustcont = true;
					break;
				}
			}
			
			if (mustcont)
				continue;
			
			mustcont = false;
			for (unsigned int h = 0; h < q->neighbors(mama->patternNodes[i]); h++) {
				bool guard = false;
				for (unsigned int k = 0; k < g->neighbors(j); k++) {				
					if (g->checkNeighborsLabels(g->adj_edges[g->adj_nodes[j] + k], q->adj_edges[q->adj_nodes[mama->patternNodes[i]] + h], q, EdgeType_NULL)) {
						guard = true;
						break;
					}				
				}
				
				if (! guard) {
					mustcont = true;
					break;
				}
			}
			
			if (mustcont)
				continue;
			
			//at position 0 there are candidates of core[0]
			candidates_v[i].push_back(j);
		}
		
		if (candidates_v[i].size() == 0)
			allpruned = true;
		
		//cout << "##: " << " Node: " << i << " Candidates: " << candidates_v[i].size() << endl;
	}
	
	if (allpruned) {
		cout << "All pruned!" << endl;
		
		delete ssol;
		delete ssol->lsolutions;
		
		return NULL;
	}
	
	//ppruning heuristic
	unsigned int powprun = 0, oldprun = 1;	
	while (oldprun != powprun) {
		oldprun = powprun;
		for (unsigned int i = 0; i < mama->nofStates; i++) {			
			unsigned int anode = mama->patternNodes[i];		
			for (unsigned int j = 0; j < q->neighbors(anode); j++) {				
				unsigned int bnode = q->adj_edges[q->adj_nodes[anode] + j];				
				powprun += this->ppruning(g, q, &candidates_v[i], &candidates_v[core_index[bnode]], mama);
			}
		}
	}
	
	ssol->length = mama->nofStates;
	ssol->candidates = new unsigned int*[mama->nofStates];	
	
	//copy from vector real candidates and calculate real solutions space	
	for (unsigned int i = 0; i < mama->nofStates; i++) {
		ssol->lsolutions[i] = candidates_v[i].size();
		
		ssol->candidates[i] = new unsigned int[ssol->lsolutions[i]];
		memcpy(ssol->candidates[i], &candidates_v[i][0], ssol->lsolutions[i] * sizeof(unsigned int));
		
		/*cout << "##: " << " Node: " << i << " Candidates: " << candidates_v[i].size() /*ssol->total_solutions << endl;
		for (int y = 0; y < candidates_v[i].size();y++)
			cout << candidates_v[i].at(y) << " ";
		cout << endl;
	*/}
	
	return ssol;
}

/*
 * ppruning heuristic function
 */
unsigned int Matcher::ppruning(Graph *g, Graph *q, vector<unsigned int> *va, vector<unsigned int> *vb, MatchingMachine *mama)
{
	unsigned int firstpp = 0;

	for (unsigned int i = 0; i < va->size(); i++) {
		bool guard = false;
		
		for (unsigned int j = 0; j < vb->size(); j++) {
			
			if (g->hasNeighbor(va->at(i), vb->at(j))) {				
				guard = true;
				break;
			}
		}
		
		if (! guard) {
			va->erase(va->begin() + i);
			firstpp++;
		}
	}
	
	return firstpp;
}
//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// End Parallel Match
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//

//
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Serial Match
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//

/*
 * Save function - Save solutions on disk in string format
 */
void save_match(ofstream *out, int *patternNodes, int *targetNodes, int length)
{
	for (int i = 0; i < length; i++) 
		*out << "(" << patternNodes[i] << "," << targetNodes[i] << ")" << endl;
		
	*out << endl;
}

/*
 * RI serial Match Algorithm - Backtracking algorithm
 */
void Matcher::match(MatchingMachine *mama, Graph *g, Graph *q)
{
	int nofStates = mama->nofStates;
		
	int *parentType = mama->parentType;
	int *parentState = mama->parentState;
	int *patternNodes = mama->patternNodes;
	int *siForPnode = mama->siForPnode;
			
	int *matchedNodes = new int[nofStates];	
	for (int i = 0; i < nofStates; i++) matchedNodes[i] = -1;
	
	int *nodeit = new int[nofStates];
	int *lastit = new int[nofStates];
	int **datait = new int*[nofStates];
	
	//filter starting node and select only real candidates
	vector<int> tempsidx_v;
	DynamicBitset *already = new DynamicBitset(g->num_nodes);
	
	for (int i = 0; i < g->num_nodes; i++)
		if (nodeMatch(g, q, i, patternNodes[0], already))
			tempsidx_v.push_back(i);

	nodeit[0] = 0;
	lastit[0] = tempsidx_v.size();
	datait[0] = new int[tempsidx_v.size()];
	memcpy(datait[0], &tempsidx_v[0], sizeof(unsigned int) * tempsidx_v.size());
	
	//cout << "Starting node of backtracking: " << tempsidx_v.size() << endl;
	
	//ofstream *out = new ofstream("./output/match.txt");
	
	int match_counter = 0;
	int psi = -1;
	int si = 0;
	int ci = -1;
	while (si != -1) {	
		if (psi >= si)
			already->set(matchedNodes[si], false);
		
		ci = -1;
		
		while (nodeit[si] < lastit[si]) {
			ci = datait[si][nodeit[si]];

			nodeit[si]++;

			matchedNodes[si] = ci;

			if (nodeMatch(g, q, ci, patternNodes[si], already) &&
			    edgeMatch(g, q, si, ci, patternNodes, siForPnode, matchedNodes)) {		
				break;
			} else {
				ci = -1;
			}			
		}		
				
		if (ci == -1) {
			psi = si;
			si--;
		} else {
			already->set(matchedNodes[si], true);
								
			if (si == nofStates - 1) {
				//save_match(out, patternNodes, matchedNodes, nofStates);
				match_counter++;
				psi = si;	
			} else {
				//fill candidate list of next state
				if (parentType[si + 1] == EdgeType_NULL) {
					
					nodeit[si + 1] = 0;
					lastit[si + 1] = g->num_nodes - 1;
					datait[si + 1] = new int[lastit[si + 1]];
					
					for (int i = 0; i < lastit[si + 1]; i++)
						datait[si + 1][i] = i;
						
					//cout << "Disconnected components!" << endl;
				} else {						
					switch (parentType[si + 1]) {
						case EdgeType_IN:
						nodeit[si + 1] = 0;
						datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].inidx];
						lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].inidx;	
						//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
						//					(unsigned int *)(&lastit[si + 1]), EdgeType_IN);
						break;
					
						case EdgeType_OUT:
						nodeit[si + 1] = 0;
						datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx];
						lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx;	
						//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
						//					(unsigned int *)(&lastit[si + 1]), EdgeType_OUT);	
						break;					
						
						default:
						nodeit[si + 1] = 0;
						datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx];
						lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].endidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx;	
						//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
						//					(unsigned int *)(&lastit[si + 1]), EdgeType_UN);
					}
					
					psi = si;
					si++;
				}				
			}
		}
	}
	
	//out->close();
	//delete out;

	cout << "Total serial match: " << match_counter << endl << endl;
	
	delete matchedNodes;	
	delete nodeit;
	delete lastit;
	delete datait[0];
	delete datait;		
	delete already;
}

/*
 * Node Match function - Checks node correctness
 */
bool Matcher::nodeMatch(Graph *g, Graph *q, int tnode, int qnode, DynamicBitset *already)
{
	//check if tnode is already matched
	if (already->get(tnode))
		return false;
	
	//check in edges
	if (g->in_nbrs[tnode] < q->in_nbrs[qnode])
		return false;
		
	//check out edges
	if (g->out_nbrs[tnode] < q->out_nbrs[qnode])
		return false;
		
	//check labels
	if (g->labels[tnode] != q->labels[qnode])
		return false;
	
	return true;
}

/*
 * Edge Match function - Checks edge correctness
 */
bool Matcher::edgeMatch(Graph *g, Graph *q, int si, int tnode, int *patternNodes, int *siForPnode, int *matchedNodes)
{		
	int pneigh;
	unsigned int *uio = &q->nbrslist[q->nbrsidx[patternNodes[si]].inidx];
	unsigned int uiolen = (unsigned int) q->nbrsidx[patternNodes[si]].outidx - q->nbrsidx[patternNodes[si]].inidx;	
	for (unsigned int i = 0; i < uiolen; i++) {
		pneigh = uio[i];
		
		if (siForPnode[pneigh] <= si)
			if (! g->hasNeighbor(matchedNodes[siForPnode[pneigh]], tnode))
				return false;	
	}
	
	uio = &q->nbrslist[q->nbrsidx[patternNodes[si]].outidx];
	uiolen = (unsigned int) q->nbrsidx[patternNodes[si]].inoutidx - q->nbrsidx[patternNodes[si]].outidx;
	for (unsigned int i = 0; i < uiolen; i++) {
		pneigh = uio[i];
		
		if (siForPnode[pneigh] <= si)
			if (! g->hasNeighbor(tnode, matchedNodes[siForPnode[pneigh]]))
				return false;
	}
	
	return true;
}

/*
 * cpu match
 */
void Matcher::cpuMatch(MatchingMachine *mama, Graph *g, Graph *q, int proc)
{
	/*vector<int> tempsidx_v;
	DynamicBitset *already = new DynamicBitset(g->num_nodes);
	
	for (int i = 0; i < g->num_nodes; i++)
		if (nodeMatch(g, q, i, mama->patternNodes[0], already))
			tempsidx_v.push_back(i);
	
	cout << "Starting node of backtracking: " << tempsidx_v.size() << endl;
	
	unsigned int thread_num = ceil((double)tempsidx_v.size() / (double)proc);
	unsigned int thread_rem = thread_num * proc - tempsidx_v.size();
	
	unsigned int mcs[proc];
	boost::thread **thr = new boost::thread*[proc];
	boost::thread_group tg;
	
	unsigned int tc = 0;
	unsigned int match_counter = 0;
	int cpus = proc;
	for (unsigned int i = 0; i < thread_num; i++) {
		
		
		if (i == thread_num - 1) {
			cpus = proc - thread_rem;
		}

		for (unsigned int k = 0; k < cpus; k++) {
			mcs[k] = 0;

			thr[k] = new boost::thread(boost::bind(&Matcher::matchThread, this, mama, g, q, tempsidx_v.at(tc++), &mcs[k]));
			tg.add_thread(thr[k]);
		}		
		
		tg.join_all();	
		
		for (unsigned int j = 0; j < cpus; j++) {
			match_counter += mcs[j];
			//delete thr[j];		
		}
	}
	
	
	
	delete thr;
	
	cout << "Total match: " << match_counter << endl;
	
	delete already;*/
} 

/*
 *
 */
void *Matcher::matchThread(MatchingMachine *mama, Graph *g, Graph *q, unsigned int start_node, unsigned int *match_counter)
{
	int nofStates = mama->nofStates;
		
	int *parentType = mama->parentType;
	int *parentState = mama->parentState;
	int *patternNodes = mama->patternNodes;
	int *siForPnode = mama->siForPnode;
			
	int *matchedNodes = new int[nofStates];	
	for (int i = 0; i < nofStates; i++) matchedNodes[i] = -1;
	
	int *nodeit = new int[nofStates];
	int *lastit = new int[nofStates];
	int **datait = new int*[nofStates];
	
	//filter starting node and select only real candidates
	vector<int> tempsidx_v;
	DynamicBitset *already = new DynamicBitset(g->num_nodes);	
	
	matchedNodes[0] = start_node;
	already->set(matchedNodes[0], true);
	
	//ofstream *out = new ofstream("./output/match.txt");
	
	int psi = -1;
	int si = 1;
	int ci = -1;
	
	si = 0;
	switch (parentType[si + 1]) {
		case EdgeType_IN:
		nodeit[si + 1] = 0;
		datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].inidx];
		lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].inidx;	
		//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
		//					(unsigned int *)(&lastit[si + 1]), EdgeType_IN);
		break;

		case EdgeType_OUT:
		nodeit[si + 1] = 0;
		datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx];
		lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx;	
		//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
		//					(unsigned int *)(&lastit[si + 1]), EdgeType_OUT);	
		break;					

		default:
		nodeit[si + 1] = 0;
		datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx];
		lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].endidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx;	
		//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
		//					(unsigned int *)(&lastit[si + 1]), EdgeType_UN);
	}
	
	si = 1;
	psi = 0;
	
	*match_counter = 0;	
	
	while (si != 0) {	
		if (psi >= si)
			already->set(matchedNodes[si], false);
		
		ci = -1;
		
		while (nodeit[si] < lastit[si]) {
			ci = datait[si][nodeit[si]];

			nodeit[si]++;

			matchedNodes[si] = ci;
			
			if (nodeMatch(g, q, ci, patternNodes[si], already) && edgeMatch(g, q, si, ci, patternNodes, siForPnode, matchedNodes)) {			
		
				break;
			} else {
				ci = -1;
			}			
		}		
				
		if (ci == -1) {
			psi = si;
			si--;
		} else {
			already->set(matchedNodes[si], true);
								
			if (si == nofStates - 1) {
				//save_match(out, patternNodes, matchedNodes, nofStates);
				*match_counter = *match_counter + 1;
				psi = si;	
			} else {
				//fill candidate list of next state
				if (parentType[si + 1] == EdgeType_NULL) {
					
					nodeit[si + 1] = 0;
					lastit[si + 1] = g->num_nodes - 1;
					datait[si + 1] = new int[lastit[si + 1]];
					
					for (int i = 0; i < lastit[si + 1]; i++)
						datait[si + 1][i] = i;		
				} else {						
					switch (parentType[si + 1]) {
						case EdgeType_IN:
						nodeit[si + 1] = 0;
						datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].inidx];
						lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].inidx;	
						//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
						//					(unsigned int *)(&lastit[si + 1]), EdgeType_IN);
						break;
					
						case EdgeType_OUT:
						nodeit[si + 1] = 0;
						datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx];
						lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].outidx;	
						//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
						//					(unsigned int *)(&lastit[si + 1]), EdgeType_OUT);	
						break;					
						
						default:
						nodeit[si + 1] = 0;
						datait[si + 1] = (int *) &g->nbrslist[g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx];
						lastit[si + 1] = (int) g->nbrsidx[matchedNodes[parentState[si + 1]]].endidx - g->nbrsidx[matchedNodes[parentState[si + 1]]].inoutidx;	
						//datait[si + 1] = (int *)g->uniqueEdges((unsigned int)matchedNodes[parentState[si + 1]],
						//					(unsigned int *)(&lastit[si + 1]), EdgeType_UN);
					}
					
					psi = si;
					si++;
				}				
			}
		}
	}
	
	//out->close();
	//delete out;
	
	delete matchedNodes;	
	delete nodeit;
	delete lastit;
	//delete datait[0];
	delete datait;		
	delete already;
	
	return NULL;
}

































//
