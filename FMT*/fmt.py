import numpy as np
from dubins import *
import matplotlib.pyplot as plt
import multiprocessing
from multiprocessing import Pool
from functools import partial

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
        self.n_cores = multiprocessing.cpu_count()
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
        self.tr_min = self.r / 10
        self.pool = Pool(processes=self.n_cores)
        # np.random.seed(0)

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
        dist = np.linalg.norm(q0[:2]-q1[:2])
        tr = np.maximum(self.tr_min,dist * 0.2)
        pts, cost = get_pts(q0,q1,tr,tr*0.1) #[y,x,theta]
        sample_num = pts.shape[0]
        for i in range(sample_num):
            if self.is_collision(pts[i,:]): # [y,x,theta]
                print(pts[i,:]*160,'col')
                return False
        return True

    def get_path(self,waypoints):
        path = np.array([[0,0,0]])
        for i in range(waypoints.shape[0] - 1):
            q0 = waypoints[i,:]
            q1 = waypoints[i+1,:]
            distance = np.linalg.norm(q0[:2]-q1[:2])
            tr = np.maximum(self.tr_min,distance * 0.2)
            pts, _ = get_pts(q0,q1,tr,tr*0.1)
            path = np.concatenate((path,pts))
        return path

    def get_neighbors(self, cand_filter, point):
        selected_idxs = self.idxs[cand_filter]
        # print("selected",selected_idxs)
        selected_points = self.points[cand_filter]
        distance_cart = np.linalg.norm(selected_points[:,:2]-point[:2],axis=1)
        # print("num cand",selected_points.shape)
        # print("selected", selected_points)
        # print("distance cart", distance_cart)
        within_rough = distance_cart < 2*self.r
        selected_idx_roughpass = selected_idxs[within_rough]
        selected_points_roughpass = selected_points[within_rough]
        # print("num rough",selected_points_roughpass.shape)
        distance = np.zeros(selected_points_roughpass.shape[0])

        ## multiprocessing
        distance_roughpass = np.linalg.norm(selected_points_roughpass[:,:2]-point[:2],axis = 1)
        tr_temp = distance_roughpass * 0.2
        tr = np.maximum(tr_temp,self.tr_min)
        infos = np.zeros((selected_points_roughpass.shape[0],5))
        infos[:,:3] = selected_points_roughpass
        infos[:,3] = tr
        infos[:,4] = tr * 0.1
        distance = self.pool.map(partial(get_cost_multi,q0=point),infos)
        distance = np.array(distance)

        ## single_thread
        # for i in range(selected_points_roughpass.shape[0]):
        #     pt = selected_points_roughpass[i]
        #     tr_temp = np.linalg.norm(pt[:2] - point[:2]) * 0.2
        #     tr = max(tr_temp,tr_min)
        #     print(tr)
        #     step_size = tr * 0.05
        #     pts1, distance[i] = get_pts(point,pt,tr,step_size)
        #     plt.plot(pts1[:,0],pts1[:,1])
        #     plt.axis('equal')
        #     print('one',pts1[0,:],pts1[-1,:])
        #     plt.show()
        #     pts2, distance[i] = get_pts(pt,point,tr,step_size)
        #     plt.plot(pts2[:,0],pts2[:,1])
        #     plt.axis('equal')
        #     print('two',pts2[0,:],pts2[-1,:])
        #     plt.show()
        within_r = distance < self.r * 5
        # err('flag')
        # print(distance_cart,"cart")
        # print(distance_roughpass)
        neighbor_idxs = selected_idx_roughpass[within_r]
        neighbor_distances = distance[within_r]
        # print("neibor distances",neighbor_distances.shape)
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
                # print("one Point found")
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
            waypoints = []
            id = self.N-1
            while id!=-1:
                waypoints.append(self.points[id,:])
                id = self.parent[id]
            path = self.get_path(np.array(waypoints))
            return path, np.array(waypoints)
        else:
            print("Plan Not found")
            return np.array([])
