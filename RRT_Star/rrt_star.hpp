#ifndef rrt_star_hpp
#define rrt_star_hpp
#include <random>
#include "tree.hpp"
#include <fstream>


class StarNode: public Node {
    public:
    double cost;
    std::vector<Node*> adj_list;
    StarNode(Point point, NodeId id);
};


class RRTStar: public Tree{
public:
    double ext_eps,sf,*map,term_th,exploit_th;
    int x_size,y_size,episodes;
    NodeId terminal_id,closest_id;
    Point start,end;
    bool is_terminal;
    // debug variables
    double min_dist, min_ee_dist;
    std::ofstream nodes,path,joints,ofs,log_;
    RRTStar(unsigned D, double* map, int x_size, int y_size);
    NodeId insert(Point pt, NodeId parent = 0) override;
    std::pair<Point,int> new_config(Point& q_near, Point& q);
    int extend(Point &q);
    void backtrack(double*** plan, int* planlength);
    bool plan(double* start, double* goal, double*** plan, int* planlength);
    void random_config(Point& q_rand, std::default_random_engine eng);
    std::vector<double> forward_kinematics(Point& angles);
    bool present(Point& pt);
    std::vector<Node*> neighborhood(Point &pt, double th);
    bool obstacle_free(Point a, Point b);
    void connect(NodeId adj_pt_id, NodeId cur_id);
};
#endif
