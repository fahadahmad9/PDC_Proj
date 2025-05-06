/*****************************************************************************
 * FILE: mo_shortest_path_mpi_non_square.c
 *
 * Naive MPI-based multi-objective (2D) shortest paths with a label-setting 
 * approach in a distributed memory setting. Each rank holds a block of nodes
 * (1D block partition). Modified to handle custom input format.
 *
 * Compilation:
 *   mpicc -O2 -o mo_shortest_path_mpi_non_square mo_shortest_path_mpi_non_square.c
 *
 * Execution:
 *   mpirun -np <num_procs> ./mo_shortest_path_mpi_non_square input.txt
 *
 * The code uses a simple bulk-synchronous approach (Allgatherv) to exchange
 * newly discovered labels (frontier items) in each iteration. It continues
 * until no new labels are discovered.
 *****************************************************************************/

 #include <stdio.h>
 #include <stdlib.h>
 #include <string.h>
 #include <stdbool.h>
 #include <mpi.h>
 #include <time.h>
 #include <assert.h>
 #include <ctype.h>
 
 /*****************************************************************************
  * Data Structures
  *****************************************************************************/
 
 /* An edge from local node 'u' to global node 'v' with a 2D cost. */
 typedef struct {
     int  v_global;  // global ID of neighbor
     int  c1;        // first objective cost
     int  c2;        // second objective cost
 } Edge;
 
 /*
  * For each local node, we store an adjacency list of Edges.
  * 'edges' are the local node's edges, 'size' is how many, etc.
  */
 typedef struct {
     Edge* edges;
     int   size;
     int   capacity;
 } EdgeList;
 
 /*
  * A label is just a 2D cost vector (c1, c2).
  */
 typedef struct {
     int c1;
     int c2;
 } Label;
 
 /*
  * Each local node can maintain multiple non-dominated labels.
  */
 typedef struct {
     Label* labels;
     int    size;
     int    capacity;
 } LabelSet;
 
 /*
  * A "frontier item" that we might need to send across MPI to another rank:
  * (global_node, c1, c2).
  */
 typedef struct {
     int node; 
     int c1;
     int c2;
 } FrontierItem;
 
 /*****************************************************************************
  * Utility Functions
  *****************************************************************************/
 
 /* Grow the capacity of an EdgeList if needed. */
 static void edgeListGrow(EdgeList* el) {
     if (el->size >= el->capacity) {
         el->capacity = (el->capacity == 0) ? 4 : el->capacity * 2;
         el->edges = (Edge*) realloc(el->edges, el->capacity * sizeof(Edge));
         if (!el->edges) {
             fprintf(stderr, "ERROR: out of memory in edgeListGrow\n");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
     }
 }
 
 /* Initialize a LabelSet. */
 static void labelSetInit(LabelSet* ls) {
     ls->labels   = NULL;
     ls->size     = 0;
     ls->capacity = 0;
 }
 
 /* Free a LabelSet. */
 static void labelSetFree(LabelSet* ls) {
     if(ls->labels) free(ls->labels);
     ls->labels   = NULL;
     ls->size     = 0;
     ls->capacity = 0;
 }
 
 /*
  * Add label (c1,c2) to LabelSet if it is not dominated.
  * Remove any labels that are dominated by the new one.
  * Return true if (c1,c2) was added, or false if it was dominated.
  */
 static bool labelSetAdd(LabelSet* ls, int c1, int c2) {
     // Check if new label is dominated by existing labels
     for(int i = 0; i < ls->size; i++) {
         int ec1 = ls->labels[i].c1;
         int ec2 = ls->labels[i].c2;
         // if existing label ec1 <= c1 and ec2 <= c2, with at least one strict:
         if(ec1 <= c1 && ec2 <= c2 && (ec1 < c1 || ec2 < c2)) {
             // new label is dominated
             return false;
         }
     }
 
     // Remove old labels that are dominated by new
     int w = 0;
     for(int i = 0; i < ls->size; i++) {
         int ec1 = ls->labels[i].c1;
         int ec2 = ls->labels[i].c2;
         bool dominatedOld = false;
         if((c1 <= ec1 && c2 <= ec2) && (c1 < ec1 || c2 < ec2)) {
             dominatedOld = true;
         }
         if(!dominatedOld) {
             ls->labels[w++] = ls->labels[i];
         }
     }
     ls->size = w;
 
     // Add the new label
     if(ls->size >= ls->capacity) {
         ls->capacity = (ls->capacity == 0) ? 4 : ls->capacity * 2;
         ls->labels = (Label*) realloc(ls->labels, ls->capacity * sizeof(Label));
         if(!ls->labels) {
             fprintf(stderr, "ERROR: out of memory in labelSetAdd\n");
             MPI_Abort(MPI_COMM_WORLD, 1);
         }
     }
     ls->labels[ls->size].c1 = c1;
     ls->labels[ls->size].c2 = c2;
     ls->size++;
 
     return true;
 }
 
 /* 
  * Given a global node index 'g', total #nodes 'n', and #ranks 'p',
  * return which rank owns node 'g' in a block distribution.
  */
 static int ownerOf(int g, int n, int p) {
     // block distribution
     int blockSize = (n + p - 1) / p; // ceiling
     int rank = g / blockSize;
     if (rank >= p) rank = p - 1; // clamp
     return rank;
 }
 
 /* Parse string to int, handling periods and whitespace */
 static int parseIntSafe(const char* str) {
     // Skip leading whitespace
     while(isspace(*str)) str++;
     
     // Check for empty string
     if(*str == '\0') return -1;
     
     // Parse the number
     char* endptr;
     int result = strtol(str, &endptr, 10);
     
     // Check if conversion was successful
     if(str == endptr) return -1;
     
     return result;
 }
 
 /*****************************************************************************
  * Reading / Distributing the Graph
  *****************************************************************************/
 
 /*
  * Modified to handle a simple list of node pairs instead of Matrix Market format
  * Each pair of integers in the file is treated as a directed edge
  */
 static void readAndDistributeGraph(const char* filename,
                                    EdgeList** localGraph, 
                                    int* localN,
                                    int* globalN,
                                    MPI_Comm comm)
 {
     int rank, size;
     MPI_Comm_rank(comm, &rank);
     MPI_Comm_size(comm, &size);
 
     // Initialize outputs
     *localGraph = NULL;
     *localN     = 0;
     *globalN    = 0;
 
     /********************************************************
      * STEP 1: Rank 0 reads the file and builds adjacency list
      ********************************************************/
     EdgeList* fullAdj = NULL; // Only allocated/used on rank 0
     
     if (rank == 0) {
         FILE* fp = fopen(filename, "r");
         if (!fp) {
             fprintf(stderr, "ERROR: Cannot open file %s\n", filename);
             MPI_Abort(comm, 1);
         }
         
         // Read first pair of numbers to determine nrows and ncols
         int nrows = 0, ncols = 0;
         if (fscanf(fp, "%d %d", &nrows, &ncols) != 2) {
             fprintf(stderr, "ERROR: Cannot read row/column counts\n");
             MPI_Abort(comm, 1);
         }
         
         // Read the third value (nnz count) but ignore it
         int dummy;
         fscanf(fp, "%d", &dummy); // Just skip past it
         
         // Use the max of nrows and ncols as the global N
         *globalN = (nrows > ncols) ? nrows : ncols;
         int N = *globalN;
         
         printf("Detected dimensions: %d rows, %d cols, using N=%d\n", nrows, ncols, N);
         
         // Allocate adjacency for all nodes
         fullAdj = (EdgeList*) malloc(N * sizeof(EdgeList));
         if (!fullAdj) {
             fprintf(stderr, "ERROR: out of memory for fullAdj\n");
             MPI_Abort(comm, 1);
         }
         
         for(int i=0; i<N; i++) {
             fullAdj[i].edges    = NULL;
             fullAdj[i].size     = 0;
             fullAdj[i].capacity = 0;
         }
         
         // Fix a seed for reproducibility
         srand(12345);
         
         // Read the rest of the file as pairs of node IDs
         char line[10240]; // Extra large to hold a full line
         
         // Read and process the remaining line(s)
         while (fgets(line, sizeof(line), fp) != NULL) {
             char* token = strtok(line, " \t\n");
             
             while (token != NULL) {
                 int src = parseIntSafe(token);
                 token = strtok(NULL, " \t\n"); // Get next token
                 
                 if (token == NULL) break; // Check if we have a pair
                 
                 int dest = parseIntSafe(token);
                 token = strtok(NULL, " \t\n"); // Move to next pair
                 
                 // Check if we have valid node IDs
                 if (src > 0 && dest > 0 && src <= N && dest <= N) {
                     // Convert to 0-based
                     src--;
                     dest--;
                     
                     // Skip self-loops
                     if (src == dest) continue;
                     
                     // Random costs
                     int c1 = rand() % 10 + 1;
                     int c2 = rand() % 10 + 1;
                     
                     // Add edge
                     edgeListGrow(&fullAdj[src]);
                     fullAdj[src].edges[fullAdj[src].size].v_global = dest;
                     fullAdj[src].edges[fullAdj[src].size].c1 = c1;
                     fullAdj[src].edges[fullAdj[src].size].c2 = c2;
                     fullAdj[src].size++;
                 }
             }
         }
         
         fclose(fp);
         
         // Debug: Print some statistics
         int totalEdges = 0;
         for (int i=0; i<N; i++) {
             totalEdges += fullAdj[i].size;
         }
         printf("Total edges read: %d\n", totalEdges);
     }
 
     /********************************************************
      * STEP 2: Broadcast globalN to all ranks
      ********************************************************/
     MPI_Bcast(globalN, 1, MPI_INT, 0, comm);
     int N = *globalN;
 
     /********************************************************
      * STEP 3: Determine local block range
      ********************************************************/
     int blockSize = (N + size - 1) / size; // ceiling
     int startNode = rank * blockSize;
     int endNode   = (rank+1) * blockSize;
     if (endNode > N) endNode = N;
     *localN = endNode - startNode;
 
     // Allocate local adjacency
     if (*localN > 0) {
         *localGraph = (EdgeList*) malloc((*localN) * sizeof(EdgeList));
         if(!(*localGraph)) {
             fprintf(stderr, "ERROR: out of memory for localGraph (rank %d)\n", rank);
             MPI_Abort(comm, 1);
         }
         for(int i=0; i<(*localN); i++) {
             (*localGraph)[i].edges    = NULL;
             (*localGraph)[i].size     = 0;
             (*localGraph)[i].capacity = 0;
         }
     }
 
     /********************************************************
      * STEP 4: Rank 0 sends each rank the adjacency data
      *         for nodes in [startNode, endNode)
      ********************************************************/
     if (rank == 0) {
         // For each rank r, gather adjacency info for [r*blockSize, min((r+1)*blockSize, N))
         for(int r=0; r<size; r++) {
             int rStart = r * blockSize;
             int rEnd   = (r+1) * blockSize;
             if (rEnd > N) rEnd = N;
             int rLocalN = rEnd - rStart;
 
             if (rLocalN <= 0) {
                 // Send a zero-length adjacency (no nodes)
                 int zero = 0;
                 MPI_Send(&zero, 1, MPI_INT, r, 999, comm);
                 continue;
             }
 
             // Prepare array of sizes (one per node)
             int* sizes = (int*) malloc(rLocalN * sizeof(int));
             if(!sizes) {
                 fprintf(stderr, "ERROR: out of memory (sizes) on rank 0\n");
                 MPI_Abort(comm, 1);
             }
 
             // Compute how many edges in total for that rank's block
             int totalEdges = 0;
             for(int node = rStart; node < rEnd; node++) {
                 int localIdx = node - rStart;
                 sizes[localIdx] = fullAdj[node].size;
                 totalEdges += fullAdj[node].size;
             }
 
             // Send the number of nodes first
             MPI_Send(&rLocalN, 1, MPI_INT, r, 999, comm);
 
             // Send the sizes array
             MPI_Send(sizes, rLocalN, MPI_INT, r, 999, comm);
 
             // Now send all edges in a contiguous buffer of (3 * totalEdges) ints
             // The format is (v_global, c1, c2) for each edge
             int* edgeBuf = (int*) malloc(3 * totalEdges * sizeof(int));
             if(!edgeBuf) {
                 fprintf(stderr, "ERROR: out of memory (edgeBuf) on rank 0\n");
                 MPI_Abort(comm, 1);
             }
             int offset = 0;
             for(int node = rStart; node < rEnd; node++) {
                 for(int e = 0; e < fullAdj[node].size; e++) {
                     edgeBuf[offset++] = fullAdj[node].edges[e].v_global;
                     edgeBuf[offset++] = fullAdj[node].edges[e].c1;
                     edgeBuf[offset++] = fullAdj[node].edges[e].c2;
                 }
             }
             // Send edgeBuf
             MPI_Send(edgeBuf, 3*totalEdges, MPI_INT, r, 999, comm);
 
             free(edgeBuf);
             free(sizes);
         }
 
         // Rank 0 also loads its own block locally
         if(*localN > 0) {
             int rLocalN = *localN; // same as endNode-startNode for rank=0
             for(int i=0; i<rLocalN; i++) {
                 int globalNode = startNode + i;
                 int sz = fullAdj[globalNode].size;
                 (*localGraph)[i].size     = sz;
                 (*localGraph)[i].capacity = sz;
                 if(sz > 0) {
                     (*localGraph)[i].edges = (Edge*) malloc(sz * sizeof(Edge));
                     if(!(*localGraph)[i].edges) {
                         fprintf(stderr, "ERROR: out of memory for edges\n");
                         MPI_Abort(comm, 1);
                     }
                     for(int e=0; e<sz; e++) {
                         (*localGraph)[i].edges[e].v_global = 
                             fullAdj[globalNode].edges[e].v_global;
                         (*localGraph)[i].edges[e].c1       = 
                             fullAdj[globalNode].edges[e].c1;
                         (*localGraph)[i].edges[e].c2       = 
                             fullAdj[globalNode].edges[e].c2;
                     }
                 }
             }
         }
 
         // Cleanup fullAdj
         if(fullAdj) {
             for(int i=0; i<N; i++) {
                 if(fullAdj[i].edges) free(fullAdj[i].edges);
             }
             free(fullAdj);
         }
     }
     else {
         // RANK != 0
         // Receive localN from rank 0
         int rLocalN = 0;
         MPI_Recv(&rLocalN, 1, MPI_INT, 0, 999, comm, MPI_STATUS_IGNORE);
         if(rLocalN != *localN) {
             // Sanity check
             if(rLocalN != 0 || *localN != 0) {
                 fprintf(stderr, "ERROR: mismatch in localN (rank %d). "
                                 "Received %d, but computed %d\n",
                         rank, rLocalN, *localN);
                 MPI_Abort(comm, 1);
             }
         }
 
         if(*localN > 0) {
             // Receive array of sizes
             int* sizes = (int*) malloc((*localN) * sizeof(int));
             if(!sizes) {
                 fprintf(stderr, "ERROR: out of memory for sizes array\n");
                 MPI_Abort(comm, 1);
             }
             MPI_Recv(sizes, *localN, MPI_INT, 0, 999, comm, MPI_STATUS_IGNORE);
 
             // Compute total edges
             int totalEdges = 0;
             for(int i=0; i<*localN; i++) {
                 totalEdges += sizes[i];
             }
 
             // Now receive the edge buffer
             int* edgeBuf = NULL;
             if(totalEdges > 0) {
                 edgeBuf = (int*) malloc(3 * totalEdges * sizeof(int));
                 if(!edgeBuf) {
                     fprintf(stderr, "ERROR: out of memory for edge buffer\n");
                     MPI_Abort(comm, 1);
                 }
                 MPI_Recv(edgeBuf, 3*totalEdges, MPI_INT, 0, 999, comm, MPI_STATUS_IGNORE);
             }
 
             // Populate localGraph
             int offset = 0;
             for(int i=0; i<*localN; i++) {
                 int sz = sizes[i];
                 (*localGraph)[i].size     = sz;
                 (*localGraph)[i].capacity = sz;
                 if(sz > 0) {
                     (*localGraph)[i].edges = (Edge*) malloc(sz * sizeof(Edge));
                     if(!(*localGraph)[i].edges) {
                         fprintf(stderr, "ERROR: out of memory for edges\n");
                         MPI_Abort(comm, 1);
                     }
                     for(int e=0; e<sz; e++) {
                         (*localGraph)[i].edges[e].v_global = edgeBuf[offset++];
                         (*localGraph)[i].edges[e].c1       = edgeBuf[offset++];
                         (*localGraph)[i].edges[e].c2       = edgeBuf[offset++];
                     }
                 }
             }
 
             if(edgeBuf) free(edgeBuf);
             free(sizes);
         }
     }
 
     // Now, each rank has *localN adjacency lists in localGraph.
     // localGraph[i] corresponds to global node (startNode + i).
 }
 
 /*****************************************************************************
  * Multi-Objective Search (Bulk-Synchronous)
  *****************************************************************************/
 static void distributedMultiObjectiveSearch(EdgeList* localGraph, 
                                             int localN, 
                                             int globalN, 
                                             int source, 
                                             MPI_Comm comm)
 {
     int rank, size;
     MPI_Comm_rank(comm, &rank);
     MPI_Comm_size(comm, &size);
 
     // Determine block distribution range for this rank
     int blockSize = (globalN + size - 1) / size;
     int startNode = rank * blockSize;
     int endNode   = (rank+1) * blockSize;
     if (endNode > globalN) endNode = globalN;
 
     // Allocate label sets for each local node
     LabelSet* labelSets = (LabelSet*) malloc(localN * sizeof(LabelSet));
     if(!labelSets && localN > 0) {
         fprintf(stderr, "ERROR: out of memory for labelSets\n");
         MPI_Abort(comm, 1);
     }
     for(int i = 0; i < localN; i++) {
         labelSetInit(&labelSets[i]);
     }
 
     // Frontier for local expansions (dynamic array)
     FrontierItem* frontier = NULL;
     int frontierSize       = 0;
     int frontierCapacity   = 0;
 
     // Helper to push an item onto the frontier
     void pushFrontier(int node, int c1, int c2) {
         if (frontierSize >= frontierCapacity) {
             frontierCapacity = (frontierCapacity == 0) ? 128 : frontierCapacity * 2;
             frontier = (FrontierItem*) realloc(frontier, 
                                               frontierCapacity * sizeof(FrontierItem));
             if (!frontier) {
                 fprintf(stderr, "ERROR: out of memory in pushFrontier\n");
                 MPI_Abort(comm, 1);
             }
         }
         frontier[frontierSize].node = node;
         frontier[frontierSize].c1   = c1;
         frontier[frontierSize].c2   = c2;
         frontierSize++;
     }
 
     // Initialize frontier with source (if this rank owns it)
     if (source >= startNode && source < endNode) {
         int localIdx = source - startNode;
         labelSetAdd(&labelSets[localIdx], 0, 0);
         pushFrontier(source, 0, 0);
     }
 
     bool done = false;
     int iteration = 0;
 
     // Bulk-synchronous iterations until no new labels
     while(!done) {
         iteration++;
         if(rank == 0 && (iteration % 10 == 0)) {
             printf("Iteration %d, frontier size: %d\n", iteration, frontierSize);
         }
 
         // 1) Exchange expansions among all ranks via Allgatherv
         int myCount = frontierSize;
         int* allCounts = (int*) malloc(size * sizeof(int));
         if(!allCounts) {
             fprintf(stderr, "ERROR: out of memory for allCounts\n");
             MPI_Abort(comm, 1);
         }
         
         MPI_Allgather(&myCount, 1, MPI_INT, allCounts, 1, MPI_INT, comm);
 
         // compute displacements
         int* displs = (int*) malloc(size * sizeof(int));
         if(!displs) {
             fprintf(stderr, "ERROR: out of memory for displs\n");
             MPI_Abort(comm, 1);
         }
         
         displs[0] = 0;
         int totalReceive = allCounts[0];
         for(int r = 1; r < size; r++) {
             displs[r] = displs[r-1] + allCounts[r-1];
             totalReceive += allCounts[r];
         }
 
         // Adjust counts and displacements for item size
         for(int r = 0; r < size; r++) {
             allCounts[r] *= 3; // each item has 3 ints
             displs[r] *= 3;
         }
 
         FrontierItem* recvBuf = NULL;
         if (totalReceive > 0) {
             recvBuf = (FrontierItem*) malloc(totalReceive * sizeof(FrontierItem));
             if (!recvBuf) {
                 fprintf(stderr, "ERROR: out of memory (recvBuf)\n");
                 MPI_Abort(comm, 1);
             }
         }
 
         // Gather the data 
         MPI_Allgatherv(frontier, myCount * 3, MPI_INT,
                       recvBuf, allCounts, displs, MPI_INT,
                       comm);
 
         // Clear our frontier for the next iteration
         frontierSize = 0;
 
         // 2) Process expansions we just received
         int newLabelsAdded = 0;
         for(int i = 0; i < totalReceive; i++) {
             int gNode = recvBuf[i].node;
             int c1    = recvBuf[i].c1;
             int c2    = recvBuf[i].c2;
 
             // If gNode is local, try to add label
             if (gNode >= startNode && gNode < endNode) {
                 int locIdx = gNode - startNode;
                 bool added = labelSetAdd(&labelSets[locIdx], c1, c2);
                 if (added) {
                     newLabelsAdded++;
                     // Expand to neighbors
                     EdgeList* eList = &localGraph[locIdx];
                     for(int ee = 0; ee < eList->size; ee++) {
                         int nbrG = eList->edges[ee].v_global;
                         int nc1  = c1 + eList->edges[ee].c1;
                         int nc2  = c2 + eList->edges[ee].c2;
                         int destRank = ownerOf(nbrG, globalN, size);
                         if(destRank == rank) {
                             // local neighbor
                             int nbrLocal = nbrG - startNode;
                             if(nbrLocal >= 0 && nbrLocal < localN) {
                                 bool added2 = labelSetAdd(&labelSets[nbrLocal], nc1, nc2);
                                 if(added2) {
                                     // enqueue for expansion
                                     pushFrontier(nbrG, nc1, nc2);
                                 }
                             }
                         } else {
                             // remote neighbor => push to frontier for next iteration
                             pushFrontier(nbrG, nc1, nc2);
                         }
                     }
                 }
             }
         }
 
         free(allCounts);
         free(displs);
         if(recvBuf) free(recvBuf);
 
         // 3) Check if frontier is empty on all ranks => done
         int localEmpty = (frontierSize == 0) ? 1 : 0;
         int globalEmpty;
         MPI_Allreduce(&localEmpty, &globalEmpty, 1, MPI_INT, MPI_MIN, comm);
         if (globalEmpty == 1) {
             done = true;
         }
     }
 
     if(rank == 0) {
         printf("Search completed after %d iterations\n", iteration);
     }
 
     /********************************************************
      * labelSets now hold the final Pareto sets for each local node.
      ********************************************************/
     // Print a small subset of local results
     if (localN > 0) {
         int limit = (localN < 5) ? localN : 5; 
         for(int i = 0; i < limit; i++) {
             int gNode = startNode + i;
             printf("[Rank %d] Node %d has %d label(s):\n", rank, gNode, labelSets[i].size);
             for(int k = 0; k < labelSets[i].size; k++) {
                 printf("   (%d, %d)\n",
                       labelSets[i].labels[k].c1,
                       labelSets[i].labels[k].c2);
             }
         }
     }
 
     // Cleanup
     if(frontier) free(frontier);
     for(int i = 0; i < localN; i++) {
         labelSetFree(&labelSets[i]);
     }
     free(labelSets);
 }
 
 /*****************************************************************************
  * MAIN
  *****************************************************************************/
 int main(int argc, char** argv)
 {
     MPI_Init(&argc, &argv);
 
     int rank, size;
     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
     MPI_Comm_size(MPI_COMM_WORLD, &size);
 
     if(argc < 2) {
         if(rank == 0) {
             fprintf(stderr, "Usage: mpirun -np <p> %s input_file.txt\n", argv[0]);
         }
         MPI_Finalize();
         return 1;
     }
 
     const char* filename = argv[1];
 
     // Read + distribute graph
     EdgeList* localGraph = NULL;
     int localN    = 0;
     int globalN   = 0;
     readAndDistributeGraph(filename, &localGraph, &localN, &globalN, MPI_COMM_WORLD);
 
     if(rank == 0) {
         printf("Graph distributed. Global nodes: %d\n", globalN);
     }
 
     // For demonstration, do multi-objective from source=0
     int source = 0;
     distributedMultiObjectiveSearch(localGraph, localN, globalN, source, MPI_COMM_WORLD);
 
     // Cleanup local adjacency
     if(localGraph) {
         for(int i = 0; i < localN; i++) {
             if(localGraph[i].edges) {
                 free(localGraph[i].edges);
             }
         }
         free(localGraph);
     }
 
     MPI_Finalize();
     return 0;
 }