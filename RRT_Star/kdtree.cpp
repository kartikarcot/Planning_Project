#include "kdtree.hpp"
#include <iostream>
//----------------------Helper Functions---------------------------//
bool is_equal(double* p1, double* p2, int N) {
  double diff = 0;
  for (int i = 0; i < N; i++) {
    diff = fabs(p1[i] - p2[i]);
    if ((diff >= __DBL_EPSILON__) && ((TWO_PI - diff) >= __DBL_EPSILON__)) {
      return false;
    }
  }
  return true;
}

void print_point(double* p, bool newline) {
  printf("(");
  for (int i = 0; i < 5; i++) {
    printf("%lf, ", p[i]);
  }
  printf(")");
  if (newline) printf("\n");
}

double sq_dist(double* p1, double* p2, int dims) {
    double total = 0.0, diff = 0;
    for (int i = 0; i < dims; i++) {
        diff = p1[i] - p2[i];
        total += (diff * diff);
    }
    return total;
}

void print_preorder_helper(KDNode* root, int depth) {
    if (root == NULL) return;
    for (int i = 0; i < depth; i++) printf("  ");
    print_point(root->val, false);
    printf(": [");
    for (KDNode* neighbor : root->adjList) {
        print_point(neighbor->val, false); printf(", ");
    }
    printf("]\n");
    for (KDNode* neighbor : root->adjList) {
        print_preorder_helper(neighbor, depth+1);
    }
}

void print_preorder(KDNode* root) {
    print_preorder_helper(root, 0);
    return;
}

//----------------------Member Functions---------------------------//
void KDTree::search_neighborhood(double* p, std::vector<NodeId>& result, double th){
    if (th < 0) {
        printf("Needs to be positive!\n");
        return;
    }
    search_neighborhood_rec(p, this->root, result, th, 0);
}

void KDTree::search_neighborhood_rec(double* p, KDNode* cur, std::vector<NodeId>& result,
        double th, int depth) {
    if (cur == NULL) return;
    // update current KDNode's distance from target for comparisons in maxheap
    cur->dist = sqrt(sq_dist(p, cur->val, this->dims));

    // add current KDNode to result if KDNode's dist is smaller than a KDNode in result
    if (cur->dist < th) {
        result.push_back(cur->id);
    }

    int split_dim = depth % this->dims;
    double split_val = cur->val[split_dim];
    double p_val = p[split_dim];
    // don't take sqrt since abs distance doesn't matter, only relative compare
    double dist_to_plane = fabs((p_val - split_val));

    // search on the side of plane that target point p lies
    if (p_val < split_val) {  // left
        search_neighborhood_rec(p, cur->left, result, th, depth+1);
    }
    else {  // right
        search_neighborhood_rec(p, cur->right, result, th, depth+1);
    }

    // if largest point in result is further than plane, need to conisder points on other side of plane, which may be closer than largest point
    // or if not enough points, search other side
    if (dist_to_plane < th) {
        // if on left side, already searched left, so search right now
        if (p_val < split_val) {
            search_neighborhood_rec(p, cur->right, result, th, depth+1);
        }
        else {
            search_neighborhood_rec(p, cur->left, result, th, depth+1);
        }
    }
}


KDTree::KDTree(KDNode* root_p, int dimsX) {
    this->root = root_p;
    this->dims = dimsX;
}

void KDTree::insert(double* p, int id_)  {
    // search for nearest neighbor before inserting p
    // so p itself is not the nearest neighbor
    // then insert and obtain ptr to this newly inserted node
    KDNode* new_node;
    this->root = insert_rec(p, id_, this->root, new_node, 0);
}

bool KDTree::search_path_reversed (double* target, 
        std::vector<KDNode*> & result) const {
    return search_path_helper(target, this->root, result);
}

bool KDTree::search_path_helper (double* target, KDNode* cur, 
        std::vector<KDNode*> & path) const {
    if (is_equal(cur->val, target, this->dims)) {
        // target will be the first elem inserted into path
        path.push_back(cur);
        return true;
    }
    for (KDNode* next : cur->adjList) {
        // previous parent calls will insert elems after target in reverse order
        if (search_path_helper(target, next, path)) {
            path.push_back(cur);
            // printf("%ld\n", path.size());
            return true;
        }
    }
    return false;
}

KDNode* KDTree::insert_rec(double* p, int id_, KDNode* cur, KDNode* & new_node, 
        int depth) {
    if (cur == NULL) {
        cur = new KDNode(p, 0, this->dims, id_);
        new_node = cur;
        return cur;
    }
    int dim = depth % this->dims;
    // if point greater than current node's split dim, search right
    if (p[dim] >= cur->val[dim]) {
        cur->right = insert_rec(p, id_, cur->right, new_node, depth+1);
    } else {  // else if less, search left
        cur->left = insert_rec(p, id_, cur->left, new_node, depth+1);
    }
    return cur;
}

void KDTree::search_knn(double* p, int k, std::vector<NodeId>& neighbors) {
    if (k < 1) {
        printf("Need to search for at least one neighbor!\n");
        return;
    }
    maxheap result;
    search_knn_rec(p, this->root, result, k, 0);
    while (!result.empty()) 
    {
        // local var on stack, but copied into neighbors so memory-safe
        // store new_p into result neighbors 
        neighbors.push_back(result.top()->id);
        result.pop();
    }
}


bool KDTree::search_graph_nn(double* p, KDNode* & nn) {
    maxheap result;
    search_knn_rec(p, this->root, result, 1, 0);
    // if found a nearest neighbor, return this
    if (result.size() > 0) {
        nn = result.top();
        return true;
    }
    else {
        nn = NULL;
        return false;
    }
}


void KDTree::search_knn_rec(double* p, KDNode* cur, maxheap & result, 
        int k, int depth) {
    if (cur == NULL) return;
    // update current KDNode's distance from target for comparisons in maxheap
    cur->dist = sq_dist(p, cur->val, this->dims);

    // add current KDNode to result if KDNode's dist is smaller than a KDNode in result
    if (result.size() < k) {
        result.push(cur);
    }
    // else if full, add if current KDNode's dist < largest dist
    else {
        if (result.top()->dist > cur->dist) {
            // remove farther point and add closer current point
            result.pop();  
            result.push(cur);  // ideally return copy, but need true ptr for non-kdtree graph connection
        }
    }

    int split_dim = depth % this->dims;
    double split_val = cur->val[split_dim];
    double p_val = p[split_dim];
    // don't take sqrt since abs distance doesn't matter, only relative compare
    double dist_to_plane = (p_val - split_val) * (p_val - split_val);

    // search on the side of plane that target point p lies
    if (p_val < split_val) {  // left
        search_knn_rec(p, cur->left, result, k, depth+1);
    }
    else {  // right
        search_knn_rec(p, cur->right, result, k, depth+1);
    }

    // if largest point in result is further than plane, need to conisder points on other side of plane, which may be closer than largest point
    // or if not enough points, search other side
    if (result.top()->dist > dist_to_plane || result.size() < k) {
        // if on left side, already searched left, so search right now
        if (p_val < split_val) {
            search_knn_rec(p, cur->right, result, k, depth+1);
        }
        else {
            search_knn_rec(p, cur->left, result, k, depth+1);
        }
    }
}
