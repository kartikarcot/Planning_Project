#!/usr/bin/env python
# -----------------------------------Packages------------------------------------------------------#
import sys
import os
import time
from enum import Enum
homedir=os.getenv("HOME")
distro=os.getenv("ROS_DISTRO")
sys.path.remove("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
import rospy
import numpy as np
import csv
from random import randint, random
import time
import pathlib
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
# msgs and services
from sampling_node.msg import Points
from sampling_node.srv import SamplingService, SamplingServiceResponse, SamplingServiceRequest
# -----------------------------------GLOBAL------------------------------------------------------#
NODE_NAME = "sampler_node"
STATE_DIM = 2
DIR = "/home/arcot/Planning_Project/src/CVAE"
MAP_NUM = None
# -----------------------------------Code------------------------------------------------------#
class Sampler(object):
    def __init__(self):
        global MAP_NUM
        self.srv = None
        MAP_NUM = rospy.get_param('/sampler/settings/map_num')
        mini_map_file = os.path.join(DIR+"/Maps/", 'map{}_mini.npy'.format(MAP_NUM))
        map_data = np.load(mini_map_file)
        self.unrolled_map = np.concatenate(map_data, axis=0)

    def initialize(self):
        global MAP_NUM
        # load tensorflow graph
        MODEL_DIR = DIR+"/Models/0"+str(MAP_NUM)
        tf.reset_default_graph()
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(MODEL_DIR+'/graph.meta')
        saver = tf.train.Saver()
        saver.restore(self.sess, tf.train.latest_checkpoint(MODEL_DIR))
        # initialize the service
        rospy.init_node(NODE_NAME)
        self.srv = rospy.Service('sampler_service', SamplingService, self.sample)

    def sample(self, req):
        z_dim = 3
        NUM_SAMPLES = req.nos  # number of samples for model to generate
        cond = np.concatenate([np.array(req.init+req.goal), self.unrolled_map])
        # same condition repeated NUM_SAMPLES times
        cond_samples = np.repeat([cond],NUM_SAMPLES,axis=0)  
        # directly sample from the latent space to generate predicted samples
        z = self.sess.graph.get_tensor_by_name('Add:0')
        c = self.sess.graph.get_tensor_by_name('c:0')
        y = self.sess.graph.get_tensor_by_name('dense_6/BiasAdd:0')
        gen_samples, _ = self.sess.run([y, z], feed_dict={z: np.random.randn(NUM_SAMPLES, z_dim), c: cond_samples})
        # construct message
        pts = Points()
        pts.D = STATE_DIM
        pts.N = NUM_SAMPLES
        pts.pts_array = gen_samples.flatten()
        res = SamplingServiceResponse(pts)
        return res

if __name__ == '__main__':
    try:
        sampler = Sampler()
        sampler.initialize()
        rospy.spin()

    except rospy.ROSInterruptException:
        pass