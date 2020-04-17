#ifndef prm_hpp
#define prm_hpp
#include <random>
#include "tree.hpp"
#include <fstream>

class PRMNode: public Node {
    public:
    std::vector<Node*> adj_list;
    double cost;
    PRMNode(Point point, NodeId id);
};
class PRM: public Tree{
public:
    double sf,*map;
    int x_size,y_size,episodes;
    Point start,end;
    bool is_terminal;
    int K;
    // debug variables
    double min_dist, min_ee_dist;
    std::ofstream nodes,path,joints,ofs;
    PRM(unsigned D, double* map, int x_size, int y_size);
    NodeId insert(Point pt, NodeId parent = 0) override;
    bool new_config(const Point& q_near, const Point& q);
    bool backtrack(double*** plan, int* planlength, int start_id, int end_id);
    bool plan(double* start, double* goal, double*** plan, int* planlength);
    std::vector<double> forward_kinematics(const Point& angles);
    bool present(const Point& pt);
    int extend(Point &q);
    void connect(NodeId adj_pt_id, NodeId cur_id);
    bool dijkstra(int s, int e, std::vector<NodeId>& res);
    std::vector<Node*> neighborhood(Point &pt, double th, int K);
};
#endif
