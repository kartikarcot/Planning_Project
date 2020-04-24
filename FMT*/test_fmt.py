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

MAP_NUM = 2
DIR = "/home/arcot/Planning_Project/src/CVAE"
MODEL_DIR = DIR+"/Models/0"+str(MAP_NUM)

class CollisionChecker(object):
    def __init__(self,_map,radius=2):
        self.radius = radius
        self._map=_map
        self._rows = _map.shape[0]
        self._cols = _map.shape[1]

    def is_in_collision(self,point):
        point_=point.astype(int)
        row_min = max(0, point_[0]-self.radius)
        row_max = min(self._rows, point_[0]+self.radius)
        col_min = max(0, point_[1]-self.radius)
        col_max = min(self._cols, point_[1]+self.radius)
        region = self._map[row_min:row_max, col_min:col_max]
        if region.any():
            return True
        return False

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

if __name__ == "__main__":
    mini_map_file = os.path.join(DIR+"/Training_Data/", 'map{}_mini.npy'.format(MAP_NUM))
    mini_map = np.load(mini_map_file)
    map_file = os.path.join(DIR+"/Training_Data/", 'map{}.npy'.format(MAP_NUM))
    _map = np.load(map_file)
    row_size, col_size = _map.shape
    # initialize the objects
    checker = CollisionChecker(_map, radius=2)
    sampler = Sampler(mini_map, row_size, col_size)
    sampler.initialize(MODEL_DIR)
    # sampled_points = sampler.sample(1000,[140,140], [10,10])
    # plt.imshow(_map)
    # plt.scatter(x=sampled_points[:,1], y=sampled_points[:,0], color='red', s=2)
    # plt.show()
    # checker.is_in_collision(np.array([140,140]))
    # a = np.array( [ 84.56912625, 120.623248])
    # b = np.array( [ 92.87981051, 120.85332846])
    planner = FMT_Star(2, 2000, sampler.sample, checker.is_in_collision)
    planner.initialize(np.array([10/160,10/160]), np.array([140/160,140/160]), np.array([0,0]), np.array([1,1]))
    path = planner.solve()
    print(path)
    plt.imshow(_map)
    plt.scatter(x=path[:,1]*160, y=path[:,0]*160, color="red", s=2)
    plt.show()