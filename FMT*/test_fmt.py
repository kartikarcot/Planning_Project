import numpy as np
from fmt import FMT_Star
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
# To suppress warnings
import warnings
warnings.filterwarnings("ignore")
import os,logging
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Tensorflow related
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

DIR = "/home/arcot/Planning_Project/src/CVAE"
# MODEL_DIR = DIR+"/Models/0"+str(MAP_NUM)

class CollisionChecker(object):
    def __init__(self,_map,radius=2):
        self.radius = radius
        self._map=_map
        self._rows = _map.shape[0]
        self._cols = _map.shape[1]

    def is_in_collision(self,point):
        point_ = np.copy(point)
        point_[0]*=self._rows
        point_[1]*=self._cols
        point_=point_.astype(int)
        row_min = max(0, point_[0]-self.radius)
        row_max = min(self._rows, point_[0]+self.radius)
        col_min = max(0, point_[1]-self.radius)
        col_max = min(self._cols, point_[1]+self.radius)
        region = self._map[row_min:row_max, col_min:col_max]
        if region.any():
            return True
        return False
#"hi"
class Sampler(object):
    def __init__(self, _map, rows, cols):
        self.unrolled_map = np.concatenate(_map, axis=0)
        self._rows = rows
        self._cols = cols
        self.sess = None

    def initialize(self, path):
        # load tensorflow graph
        tf.reset_default_graph()
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(path+'/graph.meta')
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(path))

    def sample(self, nos, init, goal):
        z_dim = 3
        NUM_SAMPLES = nos  # number of samples for model to generate
        cond = np.concatenate([np.array(init+goal), self.unrolled_map])
        # same condition repeated NUM_SAMPLES times
        cond_samples = np.repeat([cond],NUM_SAMPLES,axis=0)
        # directly sample from the latent space to generate predicted samples
        z = self.sess.graph.get_tensor_by_name('Add:0')
        c = self.sess.graph.get_tensor_by_name('c:0')
        y = self.sess.graph.get_tensor_by_name('dense_6/BiasAdd:0')
        gen_samples, _ = self.sess.run([y, z], feed_dict={z: np.random.randn(NUM_SAMPLES, z_dim), c: cond_samples})
        return gen_samples

def generate_data(_map, map_num, no_pairs=10, min_samples=20,
        filename="dataset", viz=True):
    H, W = _map.shape
    checker = CollisionChecker(_map, radius=3)
    count = 0
    data = []
    while(count!=no_pairs):
        start = np.random.random(3)*np.array([1,1,2*np.pi])
        goal = np.random.random(3)*np.array([1,1,2*np.pi])
        cond_1 = checker.is_in_collision(start)
        cond_2 = checker.is_in_collision(goal)
        if not cond_1 and not cond_2:
            try:
                checker = CollisionChecker(_map, radius=3)
                planner_linear = FMT_Star(3, 10000, None, checker.is_in_collision)
                planner_linear.initialize(start, goal, np.array([0,0,0]), np.array([1,1,2*np.pi]))
                path_lin, waypoints_lin = planner_linear.solve_linear()
                # plt.figure()
                # plt.imshow(_map)
                # plt.scatter(x=160*path_lin[:,1], y=160*path_lin[:,0], color='red', s=2)
                # plt.scatter(x=160*waypoints_lin[:,1], y=160*waypoints_lin[:,0], color='green', s=10)
                # plt.show()
                planner = FMT_Star(3, waypoints_lin.shape[0]*15, None, checker.is_in_collision)
                planner.initialize_second_pass(start, goal, np.array([0,0,0]), np.array([1,1,2*np.pi]),waypoints_lin)
                path,waypoints = planner.solve()
                path,waypoints = planner.postProcess(path,waypoints)
                print('itr:',count)
                # if path found
                if path.shape[0]!=0:
                    count+=1
                    print("%d Plans left for Map %d" % (count, map_num))
                    for item in waypoints:
                        data.append(item.tolist() + start.tolist()+ goal.tolist())

                    remaining = min_samples - len(waypoints)
                    while remaining > 0:
                        rand_wpt = None
                        remaining -= 1

                    if viz:
                        plt.figure()
                        plt.imshow(_map)
                        plt.scatter(x=W*path[:,1], y=H*path[:,0], color='red', s=2)
                        plt.scatter(x=W*waypoints[:,1], y=H*waypoints[:,0], color='green', s=10)
                        plt.show()
                        plt.close('all')
            except:
                print('error')
                pass

    np.savez(filename, data=data)

if __name__ == "__main__":
    train_val_maps = [4]  # don't train with test maps: [2, 7]
    for map_num in train_val_maps:
        map_file = os.path.join("../CVAE/Training_Data/", 'map{}.npy'.format(map_num))
        _map = np.load(map_file)
        output_file = os.path.join("../CVAE/Training_Data/", 'map{}_training'.format(map_num))
        generate_data(_map, no_pairs=300, filename=output_file, map_num=map_num, viz=False)
    print("Done!")


# if __name__ == "__main__":
#     # mini_map_file = os.path.join(DIR+"/Training_Data/", 'map{}_mini.npy'.format(MAP_NUM))
#     mini_map_file = "../CVAE/Training_Data/map2_mini.npy"
#     mini_map = np.load(mini_map_file)
#     # map_file = os.path.join(DIR+"/Training_Data/", 'map{}.npy'.format(MAP_NUM))
#     map_file = os.path.join("../CVAE/Training_Data/", 'map{}.npy'.format(MAP_NUM))
#     _map = np.load(map_file)
#     if MAP_NUM==5:
#         _map=1-_map
#     row_size, col_size = _map.shape
#     # initialize the objects
#     checker = CollisionChecker(_map, radius=3)
#     planner = FMT_Star(3, 2000, None, checker.is_in_collision)
#     planner.initialize(np.array([10/160,10/160,0]), np.array([140/160,140/160,0]), np.array([0,0,0]), np.array([1,1,2*np.pi]))
#     # comment this if you dont need to visualise the sampled points
#     # plt.scatter(x=160*planner.points[:,1], y=160*planner.points[:,0], color='red', s=2)
#     # planning the path
#     path,waypoints = planner.solve()
#     if path.shape[0]!=0:
#         plt.figure()
#         plt.imshow(_map)
#         plt.scatter(x=160*path[:,1], y=160*path[:,0], color='red', s=2)
#         plt.scatter(x=160*waypoints[:,1], y=160*waypoints[:,0], color='green', s=10)
#         plt.show()
#         plt.close('all');
