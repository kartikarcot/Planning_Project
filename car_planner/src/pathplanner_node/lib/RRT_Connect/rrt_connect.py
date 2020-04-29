from time import time
import numpy as np
from kdtree import KDTree
import matplotlib.pyplot as plt

class SimpleTree:

    def __init__(self, dim):
        self._parents_map = {}
        self._kd = KDTree(dim)

    def insert_new_node(self, point, parent=None):
        '''
        insert into kd tee
        also update the parent ma used to store the tree structure
        '''
        node_id = self._kd.insert(point)
        self._parents_map[node_id] = parent
        return node_id
        
    def get_parent(self, child_id):
        return self._parents_map[child_id]

    def get_point(self, node_id):
        '''
        get point at given node_id
        '''
        return self._kd.get_node(node_id).point

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


class RRTConnect:

    def __init__(self, dof, low, high, is_in_collision, sampler):
        self._is_in_collision = is_in_collision
        self.num_dof = dof
        self.low = low
        self.high = high
        self._connect_dist = 0.8
        self._max_n_nodes = 5000
        self._eps = 5
        self._sf = 0.01
        self._target_p = 0.5
        self._explt_th = 0.2
        self._sampler = sampler
        self.count = 0
        self.lam = 0.5
        self.sampled_points_tree = SimpleTree(self.num_dof)
        self.sampled_points = None
        self.start = None
        self.end = None
        np.random.seed(0)

    def sample_valid_points(self,point):
        '''
        insert informed sampling function here
        variate between normal and CVAE sampling
        '''
        q = None
        if np.random.random() > self._explt_th:
            # q = np.random.random(self.num_dof) * (self.high - self.low) + self.low
            if np.random.random() < self.lam:
                # q_id,_ = self.sampled_points_tree.get_nearest_node(point)
                # q = self.sampled_points_tree.get_point(q_id)
                q = self.sampled_points[self.count]
                self.count+=1
            else:
                q = np.random.random(self.num_dof) * (self.high - self.low) + self.low
        else:
            q = point
        return q

    def project_to_constraint(self, q0, constraint):
        '''
        A function to approximate a new node based on sampled nodes,
        in case the exact node is not reachabble by the controller
        '''
        return q0

    def _is_seg_valid(self, q0, q1):
        '''
        Create a trajectory and sample along the trajectory to check for collisions
        '''
        length = np.linalg.norm(q1 - q0)
        sample_num = int(length/0.5)
        qs = np.linspace(q0, q1, sample_num)
        for id,q in enumerate(qs):
            if self._is_in_collision(q):
                if id<2:
                    return 0, None
                else:
                    return 1, qs[id-1]
        return 2, q1

    
    def extend(self, tree_0, tree_1, constraint=None):
        '''
        Implement extend for RRT Connect
        - Only perform self.project_to_constraint if constraint is not None
        - Use self._is_seg_valid, self._eps, self._connect_dist
        '''
        tree=tree_0
        # q_target=tree_1.get_point
        new_node_id = None
        while True:
            q_sample = self.sample_valid_points(tree_1.root)
            near_node_id, _ = tree.get_nearest_node(q_sample)
            q_near = tree.get_point(near_node_id)
            norm = np.linalg.norm(q_sample-q_near)
            # TODO: based on the kinodynamic RRT* paper, we should not be doing this step
            q_new = q_near + min(self._eps, norm)*(q_sample-q_near)/norm
            if constraint is not None:
                q_new = self.project_to_constraint(q_new, constraint)
            signal, q_ext = self._is_seg_valid(q_near,q_new)
            if signal==0:
                continue
            # print("Found collision free configuration")
            # print(q_new)
            new_node_id = tree.insert_new_node(q_ext, parent=near_node_id)
            q1_id,_ = tree_1.get_nearest_node(q_ext)
            q1=tree_1.get_point(q1_id)
            # TODO: change this to a condition to check if collision free path is possible then connect
            connect_signal, q1_ext = self._is_seg_valid(q1,q_new)
            if connect_signal==0:
                return False, new_node_id, q1_id
            elif connect_signal==1:
                q1_new_id = tree_1.insert_new_node(q1_ext, parent=q1_id)
                return False, new_node_id, q1_new_id
            else:
                q1_new_id = tree_1.insert_new_node(q1_ext, parent=q1_id)
                return True, new_node_id, q1_new_id
            print("Did not enter if-else ladder")
            return False, new_node_id,q1_id

    def plan(self, q_start, q_target, _map, constraint=None):
        '''
        Implement the RRT Connect algorithm
        returns a list of arrays
        '''
        tree_0 = SimpleTree(len(q_start))
        tree_0.insert_new_node(q_start)
        tree_0.root = q_start
        self.start = q_start.tolist()

        tree_1 = SimpleTree(len(q_target))
        tree_1.insert_new_node(q_target)
        tree_1.root = q_target
        self.end = q_target.tolist()
        # create a sampled points tree
        self.sampled_points = self._sampler.sample(5000,self.start, self.end)
        for i in range(0,5000):
            self.sampled_points_tree.insert_new_node(self.sampled_points[i])
        # start planning
        q_start_is_tree_0 = True
        s = time()
        for n_nodes_sampled in range(self._max_n_nodes):
            if n_nodes_sampled > 0 and n_nodes_sampled % 100 == 0:
                print('RRT: Sampled {} nodes'.format(n_nodes_sampled))

            reached_target, node_id_new, node_id_1 = self.extend(tree_0, tree_1, constraint)

            if reached_target:
                break

            q_start_is_tree_0 = not q_start_is_tree_0
            tree_0, tree_1 = tree_1, tree_0

        print('RRT: Sampled {} nodes in {:.2f}s'.format(n_nodes_sampled, time() - s))

        if not q_start_is_tree_0:
            tree_0, tree_1 = tree_1, tree_0
            node_id_new, node_id_1 = node_id_1, node_id_new

        if reached_target:
            tree_0_backward_path = tree_0.construct_path_to_root(node_id_new)
            tree_1_forward_path = tree_1.construct_path_to_root(node_id_1)

            q0 = tree_0_backward_path[0]
            q1 = tree_1_forward_path[0]
            # might not need this
            # tree_01_connect_path = np.linspace(q0, q1, int(np.linalg.norm(q1 - q0) / self._sf))[1:].tolist()
            tree_01_connect_path = []
            path = tree_0_backward_path[::-1] + tree_01_connect_path + tree_1_forward_path
            print('RRT: Found a path! Path length is {}.'.format(len(path)))
        else:
            path = []
            print('RRT: Was not able to find a path!')
        
        return np.array(path)
