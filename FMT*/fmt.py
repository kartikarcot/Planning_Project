import numpy as np

MAX = 1e5

class Node:
    def __init__(self,id=None,point=None, parent=None):
        self.id = id
        self.point = point
        self.cost = 0
        self.parent = parent
        self.time = 0

class Tree:

    def __init__(self, dim):
        self._map = {}
        self.count = 0
        self.dim = dim

    def insert_new_node(self, point, parent=None):
        '''
        insert into kd tee
        also update the parent ma used to store the tree structure
        '''
        self.count+=1
        self._map[self.count] = Node(self.count, point,parent)
        return self.count
        
    def get_parent(self, child_id):
        node = self._map.get(child_id, None)
        if node is None:
            print("Not found")
            return None
        return node.parent

    def get_point(self, node_id):
        '''
        get point at given node_id
        '''
        node = self._map.get(node_id, None)
        if node is None:
            print("Not found")
            return None
        return node.point

    def get_nearest_node(self, point):
        '''
        get nodeid and distance of the nearest point to given point
        '''
        return self._kd.find_nearest_point(point)

    def construct_path_to_root(self, leaf_node_id):
        '''
        return a list of nodeids from leaf to root
        '''
        path = []
        node_id = leaf_node_id
        while node_id is not None:
            path.append(self.get_point(node_id))
            node_id = self.get_parent(node_id)
        return path

def graph_unit_test():
    g = Tree(1)
    idx = None
    for i in range(10):
        idx = g.insert_new_node(i,idx)
    path = g.construct_path_to_root(9)
    print(path)


class FMT_Star(object):
    def __init__(self, dim, N, sampler=None, is_collision=None):
        self.points = np.empty((N,dim), dtype=float)
        self.N = N
        self.num_dof = dim
        self.sampler = sampler
        self.is_collision = is_collision
        self.cost = np.full((N), 1e5, dtype=float)
        self.time = np.zeros((N), dtype=float)
        self.open = np.full((N), False, dtype=bool)
        self.closed = np.full((N), False, dtype=bool)
        self.unvisit = np.full((N), True, dtype=bool)
        self.parent = np.full((N), -1, dtype=int)
        self.idxs = np.arange(0,N,1,dtype=int)
        # TODO: Implement the empirical formula derived in paper or radius
        self.r = 0.1
        np.random.seed(0)

    def initialize(self, init, goal, low, high):
        if self.sampler is not None:
            for i in range(self.N//5):
                point = None
                while True:
                    point = np.random.random(self.num_dof) * (high - low) + low
                    if not self.is_collision(point):
                        break
                self.points[i,:] = point
            j = self.N//5
            while(True):
                generated_points = self.sampler(self.N, init.tolist(), goal.tolist())
                for i in range(self.N):
                    point = generated_points[j,:]
                    if not self.is_collision(point):
                        self.points[j,:] = point
                        j+=1
                    if j==self.N:
                        break
                if j==self.N:
                    break
            self.points[0,:] = init
            self.points[-1,:] = goal
        else:
            for i in range(self.N):
                point = None
                while True:
                    point = np.random.random(self.num_dof) * (high - low) + low
                    if not self.is_collision(point):
                        break
                self.points[i,:] = point
            self.points[0,:] = init
            self.points[-1,:] = goal
        self.open[0] = True
        self.unvisit[0] = False
        self.cost[0] = 0

    def is_seg_valid(self, q0, q1):
        # return True
        # TODO: replace with dynamic trjactory and check
        length = np.linalg.norm(q1 - q0)
        sample_num = int(length/0.005)
        qs = np.linspace(q0, q1, sample_num)
        for _,q in enumerate(qs):
            if self.is_collision(q):
                return False
        return True

    def get_neighbors(self, cand_filter, point):
        selected_idxs = self.idxs[cand_filter]
        selected_points = self.points[cand_filter]
        distance = np.linalg.norm(selected_points-point, axis=1)
        within_r = distance <self.r
        neighbor_idxs = selected_idxs[within_r]
        neighbor_distances = distance[within_r]
        return neighbor_idxs, neighbor_distances

    def extend(self):
        selected_idxs = self.idxs[self.open]
        open_list_costs = self.cost[self.open]
        cur_id = selected_idxs[open_list_costs.argmin()]
        # get neighbors in open list
        cur_unvisit_nbr_idxs, _ = self.get_neighbors(self.unvisit, self.points[cur_id])
        for unv_nbr_idx in cur_unvisit_nbr_idxs:
            # get neighbors in unvisited list
            open_nbr_idxs, open_cost = self.get_neighbors(self.open, self.points[unv_nbr_idx,:])
            net_cost = self.cost[open_nbr_idxs]+open_cost
            # select the id of the point with minimum cost
            min_cost = np.amin(net_cost)
            min_cost_idx = open_nbr_idxs[np.argmin(net_cost)]
            min_cost_point = self.points[min_cost_idx,:]
            if self.is_seg_valid(min_cost_point, self.points[unv_nbr_idx]):
                self.unvisit[unv_nbr_idx] = False
                self.parent[unv_nbr_idx] = min_cost_idx
                self.cost[unv_nbr_idx] = min_cost
                self.open[unv_nbr_idx] = True
        self.open[cur_id] = False
        self.closed[cur_id] = True

    def solve(self):
        i=0
        while(True):
            self.extend()
            i+=1
            if i%100==0:
                print(i, np.sum(self.closed))
            # break if final node visited or open list is empty
            if not self.unvisit[-1] or not self.open.any():
                break
        if not self.unvisit[-1]:
            print("Plan found")
            path = []
            id = self.N-1
            while id!=-1:
                path.append(self.points[id,:])
                id = self.parent[id]
            return np.array(path)
        else:
            print("Plan Not found")
            return np.array([])