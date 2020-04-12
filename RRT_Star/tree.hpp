//
//  kdtree.hpp
//  
//
//  Created by A S KARTHIK SAI VISHNU KUMAR on 08/03/20.
//

#ifndef tree_hpp
#define tree_hpp

#include <stdio.h>
// just for output
#include <iostream>
#include <fstream>
// stl containers
#include <unordered_map>
#include <vector>
#include <utility>
#include <set>
#include <math.h>
#include "kdtree.hpp"
//typedefs


class Node{
public:
    Point point;
    NodeId id;
    Node();
    Node(Point point,NodeId id=-1);
};

double distance(Point A, Point B);

class Comp{
public:
    Point key;
    bool reverse;
    Comp(const Point& point, bool reverse=false);
    bool operator() (const Node* A, const Node* B) const;
};


typedef std::set<Node*, Comp> pqueue;


class BoundedPQ{
public:
    pqueue* bpq;
    int k;
    BoundedPQ(int k, Point pt);
    void push(Node* node);
    void pop();
    Node* top();
};


class Tree{
public:
    unsigned D, counter;
    std::unordered_map<NodeId, NodeId> parent_map;
    std::unordered_map<NodeId, Node*> node_list;
    KDTree *kdtree;
    Tree(unsigned D);
    ~Tree();
    virtual NodeId insert(Point pt, NodeId parent=0)=0;
    Node* get_node(NodeId id);
    std::vector<Node*> get_nearest_nodes(Point &pt, int k=1);
    double get_quality(std::vector<NodeId> result);
};

#endif 
