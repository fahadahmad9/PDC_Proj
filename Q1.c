#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <time.h>  // Added for timing functionality

/*
   A simple structure to hold the edges of the graph.
   For each edge (u, v) we store:
     - v:   the neighbor index
     - c1, c2: two cost components (for demonstration).
*/
typedef struct {
    int v;
    int c1;
    int c2;
} Edge;

/*
   A dynamic array of Edge for adjacency list.
*/
typedef struct {
    Edge* edges;
    int   size;
    int   capacity;
} EdgeList;

/*
   Each label is a 2D cost vector (c1, c2).
   In a more advanced version, we might also keep path pointers, etc.
*/
typedef struct {
    int c1, c2;  // two cost components
} Label;

/*
   A node can maintain multiple non-dominated Label's.
   We'll store them in a dynamic array for demonstration.
*/
typedef struct {
    Label* labels;
    int    size;
    int    capacity;
} LabelSet;

/*
   Priority queue node for expansions: we pick next label to expand
   by some priority, e.g., minimal c1 + c2. For demonstration,
   we will push all labels in a simple array-based structure
   or a min-heap, and pop in ascending order of c1 + c2.
*/
typedef struct {
    int   node;
    int   c1;
    int   c2;
} PQItem;

/* A simple dynamic array for PQ items. */
typedef struct {
    PQItem* data;
    int     size;
    int     capacity;
} PQ;

/* Forward declarations */
void pq_init(PQ* pq);
void pq_push(PQ* pq, int node, int c1, int c2);
bool pq_pop(PQ* pq, PQItem* out);
void pq_free(PQ* pq);

////////////////////////////////////////////
// Helper: readMatrixMarketGraph
////////////////////////////////////////////
/*
   Reads a "pattern" Matrix Market file for an undirected graph.
   - Skips comment lines starting with '%' or '%%'
   - First non-comment line has 3 integers: #rows, #cols, #nnz
   - Next nnz lines each contain 2 integers (i, j)
   - We assume 1-based indices in the file, and we convert them to 0-based internally.
   - We build an adjacency list for each of the #rows= #cols vertices.

   For demonstration, we randomly assign two cost components c1, c2 to each edge.
*/
EdgeList* readMatrixMarketGraph(const char* filename, int* nOut) {
    FILE* fp = fopen(filename, "r");
    if(!fp) {
        fprintf(stderr, "ERROR: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Skip comment lines
    char line[1024];
    while(true) {
        long pos = ftell(fp);
        if(!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "ERROR: Unexpected end of file\n");
            exit(EXIT_FAILURE);
        }
        if(line[0] == '%' || line[0] == '\0') {
            continue; // skip
        }
        // else we revert one line so we can parse it as the dimension line
        fseek(fp, pos, SEEK_SET);
        break;
    }

    // Read nrows, ncols, nnz from the first non-comment line
    int nrows, ncols;
    long long nnz;
    if(fscanf(fp, "%d %d %lld", &nrows, &ncols, &nnz) != 3) {
        fprintf(stderr, "ERROR: Invalid matrix header\n");
        exit(EXIT_FAILURE);
    }

    // We assume nrows == ncols
    int n = nrows;
    *nOut = n;

    // Allocate adjacency list
    EdgeList* graph = (EdgeList*) malloc(n * sizeof(EdgeList));
    if(!graph) {
        fprintf(stderr, "ERROR: Out of memory (graph)\n");
        exit(EXIT_FAILURE);
    }

    for(int i = 0; i < n; i++) {
        graph[i].edges    = NULL;
        graph[i].size     = 0;
        graph[i].capacity = 0;
    }

    // A helper macro to grow adjacency list
    #define EL_GROW(el) do {                                            \
        if((el).size >= (el).capacity) {                                \
            (el).capacity = ((el).capacity == 0) ? 4 : (el).capacity*2; \
            (el).edges = (Edge*) realloc((el).edges, (el).capacity*sizeof(Edge)); \
            if(!(el).edges) {                                           \
                fprintf(stderr, "ERROR: Out of memory (edges)\n");      \
                exit(EXIT_FAILURE);                                     \
            }                                                           \
        }                                                               \
    } while(0)

    // Read edges
    for(long long e = 0; e < nnz; e++) {
        int i, j;
        if(fscanf(fp, "%d %d", &i, &j) != 2) {
            fprintf(stderr, "ERROR: Invalid edge line\n");
            exit(EXIT_FAILURE);
        }
        // Convert 1-based to 0-based
        i--; 
        j--;

        if(i == j) {
            // Typically diagonal for a symmetric matrix is irrelevant
            // or self-loop. We can ignore or handle specially.
            continue;
        }

        // Random 2D cost for demonstration:
        int c1 = rand() % 10 + 1;  // random cost from 1..10
        int c2 = rand() % 10 + 1;

        // Add edge i->j
        EL_GROW(graph[i]);
        graph[i].edges[ graph[i].size ].v  = j;
        graph[i].edges[ graph[i].size ].c1 = c1;
        graph[i].edges[ graph[i].size ].c2 = c2;
        graph[i].size++;

        // Since undirected, add edge j->i
        EL_GROW(graph[j]);
        graph[j].edges[ graph[j].size ].v  = i;
        graph[j].edges[ graph[j].size ].c1 = c1;
        graph[j].edges[ graph[j].size ].c2 = c2;
        graph[j].size++;
    }

    fclose(fp);
    return graph;
}

////////////////////////////////////////////
// Multi-objective data structures
////////////////////////////////////////////

/* Initialize a LabelSet */
void labelset_init(LabelSet* ls) {
    ls->labels   = NULL;
    ls->size     = 0;
    ls->capacity = 0;
}

/* Free a LabelSet */
void labelset_free(LabelSet* ls) {
    if(ls->labels) free(ls->labels);
    ls->labels   = NULL;
    ls->size     = 0;
    ls->capacity = 0;
}

/* Add label (c1, c2) to LabelSet, removing dominated labels.
   Returns true if it was actually added, false if it was dominated (and thus not added).
*/
bool labelset_add(LabelSet* ls, int c1, int c2) {
    // First, check if the new label is dominated by any existing label
    for(int i = 0; i < ls->size; i++) {
        int ec1 = ls->labels[i].c1;
        int ec2 = ls->labels[i].c2;
        // If existing label ec1 <= c1, ec2 <= c2, and at least one is strictly <, we are dominated
        if(ec1 <= c1 && ec2 <= c2 && (ec1 < c1 || ec2 < c2)) {
            return false; // new label is dominated -> do not add
        }
    }

    // If not dominated, we might dominate some existing ones, so remove them
    // We can do this in a second pass
    int writeIdx = 0;
    for(int i = 0; i < ls->size; i++) {
        int ec1 = ls->labels[i].c1;
        int ec2 = ls->labels[i].c2;
        // If our new label dominates an existing label, we skip that existing label.
        // new label dominates if c1 <= ec1, c2 <= ec2, and (c1 < ec1 or c2 < ec2)
        bool dominatedOld = false;
        if((c1 <= ec1 && c2 <= ec2) && (c1 < ec1 || c2 < ec2)) {
            dominatedOld = true;
        }
        if(!dominatedOld) {
            // keep the old label
            ls->labels[writeIdx++] = ls->labels[i];
        }
    }
    ls->size = writeIdx;

    // Finally add the new label
    if(ls->size >= ls->capacity) {
        ls->capacity = (ls->capacity == 0) ? 4 : ls->capacity * 2;
        ls->labels = (Label*) realloc(ls->labels, ls->capacity * sizeof(Label));
        if(!ls->labels) {
            fprintf(stderr, "ERROR: out of memory in labelset_add\n");
            exit(EXIT_FAILURE);
        }
    }
    ls->labels[ls->size].c1 = c1;
    ls->labels[ls->size].c2 = c2;
    ls->size++;

    return true;
}

////////////////////////////////////////////
// Priority queue for expansions
////////////////////////////////////////////

void pq_init(PQ* pq) {
    pq->data = NULL;
    pq->size = 0;
    pq->capacity = 0;
}

/* Reserve space in PQ dynamic array */
void pq_reserve(PQ* pq, int newcap) {
    if(newcap > pq->capacity) {
        pq->data = (PQItem*) realloc(pq->data, newcap * sizeof(PQItem));
        pq->capacity = newcap;
        if(!pq->data) {
            fprintf(stderr, "ERROR: out of memory in pq_reserve\n");
            exit(EXIT_FAILURE);
        }
    }
}

/* Insert an item in the simplest manner:
   We do NOT do a fancy binary-heap here. Instead, we'll store them all
   and do a linear search on pop. This is O(n^2) in worst case but simpler
   for demonstration. Feel free to implement a real min-heap for large graphs.
*/
void pq_push(PQ* pq, int node, int c1, int c2) {
    if(pq->size >= pq->capacity) {
        int newcap = (pq->capacity == 0) ? 1024 : pq->capacity * 2;
        pq_reserve(pq, newcap);
    }
    pq->data[pq->size].node = node;
    pq->data[pq->size].c1   = c1;
    pq->data[pq->size].c2   = c2;
    pq->size++;
}

/* Pop item with smallest (c1 + c2). Returns false if empty. */
bool pq_pop(PQ* pq, PQItem* out) {
    if(pq->size == 0) return false;

    // find min by c1 + c2
    int bestIdx = 0;
    int bestSum = pq->data[0].c1 + pq->data[0].c2;
    for(int i = 1; i < pq->size; i++) {
        int s = pq->data[i].c1 + pq->data[i].c2;
        if(s < bestSum) {
            bestSum = s;
            bestIdx = i;
        }
    }

    // Return that item
    *out = pq->data[bestIdx];

    // Move last item to fill the gap
    pq->size--;
    pq->data[bestIdx] = pq->data[pq->size];

    return true;
}

void pq_free(PQ* pq) {
    if(pq->data) free(pq->data);
    pq->data = NULL;
    pq->size = 0;
    pq->capacity = 0;
}

////////////////////////////////////////////
// Multi-objective shortest path (2D)
////////////////////////////////////////////
/*
   Given an adjacency list graph, run multi-objective label-setting from
   a single source (0 by default, or choose a different source).
   We maintain a LabelSet for each node, as well as a priority queue
   for expansions.
*/
void multiObjectiveSP(EdgeList* graph, int n, int source, LabelSet* labelSets) {
    // Initialize labelSets
    for(int i = 0; i < n; i++) {
        labelset_init(&labelSets[i]);
    }

    // Insert the initial label (0,0) for source
    labelset_add(&labelSets[source], 0, 0);

    // Priority queue
    PQ pq;
    pq_init(&pq);

    // Push (source, 0, 0) into PQ
    pq_push(&pq, source, 0, 0);

    // Expand
    PQItem current;
    while(pq_pop(&pq, &current)) {
        int   u  = current.node;
        int   uc1 = current.c1;
        int   uc2 = current.c2;

        // Because multiple labels for u may have been added after we queued this label,
        // we should check that (uc1, uc2) is still present (not dominated) in labelSets[u].
        // If it's gone (dominated), skip expansion.
        bool stillValid = false;
        for(int i = 0; i < labelSets[u].size; i++) {
            if(labelSets[u].labels[i].c1 == uc1 &&
               labelSets[u].labels[i].c2 == uc2) {
                stillValid = true;
                break;
            }
        }
        if(!stillValid) continue; // skip, we are dominated

        // For each neighbor v of u
        for(int i = 0; i < graph[u].size; i++) {
            int v   = graph[u].edges[i].v;
            int ec1 = graph[u].edges[i].c1;
            int ec2 = graph[u].edges[i].c2;
            // new cost if we go from source->u plus edge (u,v)
            int newc1 = uc1 + ec1;
            int newc2 = uc2 + ec2;

            // Try to add label (newc1, newc2) to labelSets[v]
            bool added = labelset_add(&labelSets[v], newc1, newc2);
            if(added) {
                // If we successfully added a new non-dominated label, push to PQ
                pq_push(&pq, v, newc1, newc2);
            }
        }
    }

    pq_free(&pq);
}


int main(int argc, char* argv[]) {
    if(argc < 2) {
        fprintf(stderr, "Usage: %s <matrix_market_file>\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Set up timing
    clock_t start, end;
    double cpu_time_used;
    
    // read the graph
    int n = 0;
    printf("Reading graph from file: %s\n", argv[1]);
    EdgeList* graph = readMatrixMarketGraph(argv[1], &n);
    
    // Print the number of nodes
    printf("Graph size (n): %d nodes\n", n);

    // for demonstration, we'll fix source = 0
    int source = 0;

    // allocate labelSets for all nodes
    LabelSet* labelSets = (LabelSet*) malloc(n * sizeof(LabelSet));
    if(!labelSets) {
        fprintf(stderr, "ERROR: out of memory (labelSets)\n");
        return EXIT_FAILURE;
    }

    // Start timing the algorithm execution
    printf("Starting multi-objective shortest path algorithm...\n");
    start = clock();
    
    // run multi-objective shortest path
    multiObjectiveSP(graph, n, source, labelSets);
    
    // End timing
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Algorithm execution time: %.6f seconds\n\n", cpu_time_used);

    // Count total number of labels generated
    int totalLabels = 0;
    for(int i = 0; i < n; i++) {
        totalLabels += labelSets[i].size;
    }
    printf("Total labels generated: %d (avg %.2f per node)\n\n", 
           totalLabels, (float)totalLabels/n);

    // Print out a sample of the Pareto sets for demonstration.
    // We'll just show the first few nodes, since n can be very large.
    int limit = (n < 100) ? n : 100;
    printf("Sample of Pareto fronts for first %d nodes:\n", limit);
    for(int i = 0; i < limit; i++) {
        printf("Node %d: %d label(s)\n", i, labelSets[i].size);
        for(int k = 0; k < labelSets[i].size; k++) {
            printf("   (%d, %d)\n", labelSets[i].labels[k].c1,
                                    labelSets[i].labels[k].c2);
        }
    }

    // Cleanup
    for(int i = 0; i < n; i++) {
        free(graph[i].edges);
        labelset_free(&labelSets[i]);
    }
    free(graph);
    free(labelSets);

    return 0;
}