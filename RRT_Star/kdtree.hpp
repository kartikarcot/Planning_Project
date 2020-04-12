#ifndef __KDTREE__
#define __KDTREE__

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>
#include <queue>
#include <math.h>
#include <vector>
// #include "helpers.hpp"

// Places to Optimize Code:
// currently create KDNodes to insert into result, but only use one data member of KDNode, so wasteful... instead just insert point as a vector
// DIMS is currently a constant, make adaptable by passing dynamically allocated arrays or vectors to handle arbitrary dimensions for arms
// Reference: https://github.com/stefankoegl/kdtree/blob/587edc7056d7735177ad56a84ad5abccdea91693/kdtree.py#L187
typedef int NodeId;
typedef std::vector<double> Point;
const double TWO_PI = 2 * M_PI; 
struct KDNode {
    KDNode* left = NULL;
    KDNode* right = NULL;
    std::vector<KDNode*> adjList;  // graph connections
    int dims;
    double* val;
    double dist = 0.0;  // distance from current search point, old value invalid so need to update when entering new KDNode into pq
    int id;
    // ensures won't forget to pass updated distance for comparisons
    KDNode(double* valX, double distX, int dimsX, int id_=-1) {
        val = (double*)malloc(dimsX * sizeof(double));
        for (int i = 0; i < dimsX; i++) this->val[i] = valX[i];
        this->dist = distX;
        this->dims = dimsX;
        id=id_;
    }
};


class KDNodeCompare
// MaxHeap: compare distance from target of one neighbor to another 
{
public:
    bool operator() (const KDNode* cur, const KDNode* other)
    {
        // true(sift up for maxHeap) if other is farther away
        return cur->dist < other->dist;
    }
};

typedef std::priority_queue<KDNode*, std::vector<KDNode*>, KDNodeCompare> maxheap;
typedef std::vector<std::vector<double>> pts_list;

class KDTree {
    KDTree();  // prevent default constructor
public:
    KDNode* root = NULL;
    int dims = 0;
    KDTree(KDNode* root_p, int dimsX);
    void insert(double* p, int id_);
    KDNode* insert_rec(double* p, int id_, KDNode* cur, KDNode* & new_node, 
        int depth);
    void search_knn(double* p, int k, std::vector<NodeId>& neighbors);
    void search_knn_rec(double* p, KDNode* cur, maxheap & result, 
        int k, int d);
    bool search_graph_nn(double* p, KDNode* & nn);
    bool search_path_reversed(double* target, 
        std::vector<KDNode*> & result) const;
    bool search_path_helper(double* target, KDNode* cur, 
        std::vector<KDNode*> & path) const;
    void search_neighborhood(double* p, std::vector<NodeId>& result, double th);
    void search_neighborhood_rec(double* p, KDNode* cur, std::vector<NodeId>& result,
                                double th, int depth);
};

#endif
