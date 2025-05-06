#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <stdbool.h>
#include <omp.h>  // For OpenMP parallel processing

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
   A dynamic array of Edge for adjacency lists.
*/
typedef struct {
    Edge* edges;
    int   size;
    int   capacity;
} EdgeList;

/*
   Each label is a 2D cost vector (c1, c2).
   We might also store path pointers, etc. in more advanced versions.
*/
typedef struct {
    int c1, c2;
} Label;

/*
   A node can have multiple non-dominated Label's.
   We'll store them in a dynamic array.
*/
typedef struct {
    Label* labels;
    int    size;
    int    capacity;
} LabelSet;

/*
   Priority queue item for expansions.
   We'll store (node, c1, c2).
*/
typedef struct {
    int   node;
    int   c1;
    int   c2;
} PQItem;

/*
   A simple dynamic array for PQ items.
   We'll do a linear search on pop() to find the min (c1+c2).
*/
typedef struct {
    PQItem* data;
    int     size;
    int     capacity;
} PQ;

/* Forward declarations for PQ */
void pq_init(PQ* pq);
void pq_push(PQ* pq, int node, int c1, int c2);
bool pq_pop(PQ* pq, PQItem* out);
void pq_free(PQ* pq);

////////////////////////////////////////////////////////////
// Helper: readMatrixMarketGraph
////////////////////////////////////////////////////////////
/*
   Reads a "pattern" Matrix Market file for an undirected graph.
   - Skips comment lines starting with '%' or '%%'
   - First non-comment line has 3 integers: #rows, #cols, #nnz
   - Next nnz lines each contain 2 integers (i, j)
   - We assume 1-based indices in the file, and convert them to 0-based internally.
   - We build an adjacency list for each of the n (= rows=cols) vertices.

   For demonstration, we randomly assign two cost components (c1, c2) to each edge.
*/
EdgeList* readMatrixMarketGraph(const char* filename, int* nOut) {
    FILE* fp = fopen(filename, "r");
    if(!fp) {
        fprintf(stderr, "ERROR: Cannot open file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    // Skip initial comment lines
    char line[1024];
    while(true) {
        long pos = ftell(fp);
        if(!fgets(line, sizeof(line), fp)) {
            fprintf(stderr, "ERROR: Unexpected end of file\n");
            exit(EXIT_FAILURE);
        }
        if(line[0] == '%' || line[0] == '\0') {
            continue; // skip comment
        }
        // Rewind one line so we parse it as dimension
        fseek(fp, pos, SEEK_SET);
        break;
    }

    // Read dimension line: nrows, ncols, nnz
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

        // If i == j, it's a self-loop (ignore or handle specially). We'll ignore here.
        if(i == j) {
            continue;
        }

        // Random 2D cost
        int c1 = rand() % 10 + 1; // random 1..10
        int c2 = rand() % 10 + 1;

        // Add edge i->j
        EL_GROW(graph[i]);
        graph[i].edges[ graph[i].size ].v  = j;
        graph[i].edges[ graph[i].size ].c1 = c1;
        graph[i].edges[ graph[i].size ].c2 = c2;
        graph[i].size++;

        // Undirected: add edge j->i
        EL_GROW(graph[j]);
        graph[j].edges[ graph[j].size ].v  = i;
        graph[j].edges[ graph[j].size ].c1 = c1;
        graph[j].edges[ graph[j].size ].c2 = c2;
        graph[j].size++;
    }

    fclose(fp);
    return graph;
}

////////////////////////////////////////////////////////////
// LabelSet management
////////////////////////////////////////////////////////////

void labelset_init(LabelSet* ls) {
    ls->labels   = NULL;
    ls->size     = 0;
    ls->capacity = 0;
}

void labelset_free(LabelSet* ls) {
    if(ls->labels) free(ls->labels);
    ls->labels   = NULL;
    ls->size     = 0;
    ls->capacity = 0;
}

/*
   Add label (c1, c2) to LabelSet ls, removing dominated labels.
   Returns true if (c1,c2) was added, false if it was found to be dominated.
   We wrap this in a #pragma omp critical to ensure thread safety.
*/
bool labelset_add(LabelSet* ls, int c1, int c2) {
    // First check: is the new label dominated by an existing label?
    for(int i = 0; i < ls->size; i++) {
        int ec1 = ls->labels[i].c1;
        int ec2 = ls->labels[i].c2;
        // If existing label ec1 <= c1 and ec2 <= c2, with at least one strict:
        if(ec1 <= c1 && ec2 <= c2 && (ec1 < c1 || ec2 < c2)) {
            return false; // new label dominated
        }
    }

    // If not dominated, remove labels that are dominated by the new one
    int writeIdx = 0;
    for(int i = 0; i < ls->size; i++) {
        int ec1 = ls->labels[i].c1;
        int ec2 = ls->labels[i].c2;
        bool dominatedOld = false;
        if((c1 <= ec1 && c2 <= ec2) && (c1 < ec1 || c2 < ec2)) {
            dominatedOld = true;
        }
        if(!dominatedOld) {
            ls->labels[writeIdx++] = ls->labels[i];
        }
    }
    ls->size = writeIdx;

    // Add the new label
    if(ls->size >= ls->capacity) {
        ls->capacity = (ls->capacity == 0) ? 4 : (ls->capacity * 2);
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

////////////////////////////////////////////////////////////
// Priority Queue
////////////////////////////////////////////////////////////

void pq_init(PQ* pq) {
    pq->data = NULL;
    pq->size = 0;
    pq->capacity = 0;
}

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

/*
   Insert in simplest manner: append and rely on a linear search in pop().
   We'll protect pushes with a critical section to ensure thread safety.
*/
void pq_push(PQ* pq, int node, int c1, int c2) {
    #pragma omp critical (PQPush)
    {
        if(pq->size >= pq->capacity) {
            int newcap = (pq->capacity == 0) ? 1024 : pq->capacity * 2;
            pq_reserve(pq, newcap);
        }
        pq->data[pq->size].node = node;
        pq->data[pq->size].c1   = c1;
        pq->data[pq->size].c2   = c2;
        pq->size++;
    }
}

/*
   Pop item with smallest (c1 + c2) via linear search.
   If empty, return false. We also protect this with a critical section.
*/
bool pq_pop(PQ* pq, PQItem* out) {
    #pragma omp critical (PQPop)
    {
        if(pq->size == 0) {
            out->node = -1;
            out->c1   = -1;
            out->c2   = -1;
        } else {
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
            // Move last to fill
            pq->size--;
            pq->data[bestIdx] = pq->data[pq->size];
        }
    }
    return (out->node != -1);
}

void pq_free(PQ* pq) {
    if(pq->data) free(pq->data);
    pq->data = NULL;
    pq->size = 0;
    pq->capacity = 0;
}

////////////////////////////////////////////////////////////
// Multi-objective shortest path (2D) with naive OpenMP
////////////////////////////////////////////////////////////
void multiObjectiveSP(EdgeList* graph, int n, int source, LabelSet* labelSets) {
    // Initialize labelSets
    #pragma omp parallel for
    for(int i = 0; i < n; i++) {
        labelset_init(&labelSets[i]);
    }

    // Insert the initial label (0,0) for source
    labelset_add(&labelSets[source], 0, 0);

    // Priority queue
    PQ pq;
    pq_init(&pq);

    // Push initial
    pq_push(&pq, source, 0, 0);

    // We'll pop items one by one; after popping, we do a parallel expansion
    while(true) {
        PQItem current;
        bool hasItem = pq_pop(&pq, &current);
        if(!hasItem) {
            // PQ empty -> done
            break;
        }

        int u = current.node;
        int uc1 = current.c1;
        int uc2 = current.c2;

        // Check if (uc1, uc2) is still valid in labelSets[u] (not dominated)
        bool stillValid = false;
        {
            LabelSet* ls = &labelSets[u];
            // We don't strictly need a lock to read here if we assume no one modifies
            // labelSets[u] concurrently, but to be safe:
            #pragma omp critical (LabelSetCheck)
            {
                for(int i = 0; i < ls->size; i++) {
                    if(ls->labels[i].c1 == uc1 && ls->labels[i].c2 == uc2) {
                        stillValid = true;
                        break;
                    }
                }
            }
        }
        if(!stillValid) {
            // This label was dominated after it was queued -> skip
            continue;
        }

        // Expand to neighbors in parallel
        int deg = graph[u].size;
        #pragma omp parallel for
        for(int i = 0; i < deg; i++) {
            int v   = graph[u].edges[i].v;
            int ec1 = graph[u].edges[i].c1;
            int ec2 = graph[u].edges[i].c2;

            int newc1 = uc1 + ec1;
            int newc2 = uc2 + ec2;

            // Safely add label to labelSets[v]
            bool addedLabel = false;
            #pragma omp critical (LabelSetAdd)
            {
                addedLabel = labelset_add(&labelSets[v], newc1, newc2);
            }

            if(addedLabel) {
                // If we successfully added a new label, queue it
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

    // Set up timing using OpenMP's high precision timer
    double start_time, end_time;
    
    // Read the graph
    int n = 0;
    printf("Reading graph from file: %s\n", argv[1]);
    EdgeList* graph = readMatrixMarketGraph(argv[1], &n);
    
    // Print the number of nodes and threads
    printf("Graph size (n): %d nodes\n", n);
    printf("Number of OpenMP threads: %d\n", omp_get_max_threads());

    // We'll choose source = 0 for demonstration
    int source = 0;

    // Allocate labelSets
    LabelSet* labelSets = (LabelSet*) malloc(n * sizeof(LabelSet));
    if(!labelSets) {
        fprintf(stderr, "ERROR: out of memory (labelSets)\n");
        return EXIT_FAILURE;
    }

    // Start timing the algorithm execution
    printf("Starting multi-objective shortest path algorithm (OpenMP version)...\n");
    start_time = omp_get_wtime();
    
    // Run multi-objective shortest path
    multiObjectiveSP(graph, n, source, labelSets);
    
    // End timing
    end_time = omp_get_wtime();
    printf("Algorithm execution time: %.6f seconds\n\n", end_time - start_time);

    // Count total number of labels generated
    int totalLabels = 0;
    for(int i = 0; i < n; i++) {
        totalLabels += labelSets[i].size;
    }
    printf("Total labels generated: %d (avg %.2f per node)\n\n", 
           totalLabels, (float)totalLabels/n);

    // Print the Pareto sets for the first few nodes
    int limit = (n < 20) ? n : 20;
    printf("Sample of Pareto fronts for first %d nodes:\n", limit);
    for(int i = 0; i < limit; i++) {
        printf("Node %d: %d label(s)\n", i, labelSets[i].size);
        for(int k = 0; k < labelSets[i].size; k++) {
            printf("   (%d, %d)\n",
                   labelSets[i].labels[k].c1,
                   labelSets[i].labels[k].c2);
        }
    }

    // Cleanup
    for(int i = 0; i < n; i++) {
        if(graph[i].edges) free(graph[i].edges);
        labelset_free(&labelSets[i]);
    }
    free(graph);
    free(labelSets);

    return 0;
}