import os
import time
import sys
homedir=os.getenv("HOME")
distro=os.getenv("ROS_DISTRO")
sys.path.remove("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
sys.path.append("/opt/ros/"+distro+"/lib/python2.7/dist-packages")
import rospy
from sampling_node.msg import Points
from sampling_node.srv import SamplingService, SamplingServiceResponse, SamplingServiceRequest

if __name__ == "__main__":
    rospy.wait_for_service('sampler_service')
    sampler = rospy.ServiceProxy('sampler_service',SamplingService)
    req = SamplingServiceRequest()
    req.nos = 2
    req.init = [0.1,0.1]
    req.goal = [0.2,0.1]
    resp = sampler(req)
    print(resp)