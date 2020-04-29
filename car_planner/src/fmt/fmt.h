//
// Created by tom on 4/28/20.
//

#ifndef KINODYNAMIC_RRT_PLANNER_FMT_H
#define KINODYNAMIC_RRT_PLANNER_FMT_H

#include "KDTree/src/KDTree.h"
//#include "KDTree/src/Point.h"
#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

class FmtNode {
public:
    FmtNode(int id, vector<double> point, shared_ptr<FmtNode> parent) {

    }
    int id;
    vector<double> point;

};

#endif //KINODYNAMIC_RRT_PLANNER_FMT_H
