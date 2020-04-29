import rospy
import tf
from std_msgs.msg import Header
from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, PoseStamped, Point, PointStamped
from nav_msgs.srv import GetPlan, GetPlanResponse
from abc import ABCMeta, abstractmethod
import yaml
from PIL import Image
import numpy as np

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

DEBUG_PRINT = True
GRID_OFFSET = 0  # set to 0.5 for cell-edge indexing


def dbg(args):
    if DEBUG_PRINT:
        rospy.loginfo(args)


class RobotShape:
    __metaclass__ = ABCMeta

    @abstractmethod
    def footprint(self, pos, resolution):
        # should return a list of points that are under the footprint
        pass

    @abstractmethod
    def map_expansion(self, resolution):
        # should return an int describing how much map expansion to use. (zero means none)
        # if the map is pre-expanded, then leave this as zero.
        pass


class CircleRobot(RobotShape):
    def __init__(self, radius):
        pass


class MapService:
    def __init__(self, map_directory, robot):
        self.map = None
        self.map_expanded = None
        self.robot = robot
        self.map_pub = rospy.Publisher('map', OccupancyGrid, queue_size=10)
        self.load_map_object(map_directory)

    def load_map_object(self, map_directory):
        yaml_path = map_directory + "/map.yaml"
        try:
            yaml_file = file(yaml_path, 'r')
        except IOError:
            rospy.logerr('Could not find map yaml!')
            return
        yaml_data = yaml.load(yaml_file, Loader=Loader)
        imgname = yaml_data['image']
        img = Image.open(map_directory + '/' + imgname)
        mapdata = to_grayscale_array(img)

        thresh = yaml_data['thresh']
        thresholded = map(lambda x: [100 if x < 255*thresh else 0 for x in x], mapdata)

        thresholded = list(np.array(thresholded).transpose())

        g = OccupancyGrid()
        g.header.stamp = rospy.Time.now()
        g.header.frame_id = yaml_data['frame']
        g.info.map_load_time = rospy.Time.now()
        g.info.resolution = yaml_data['resolution']
        g.info.width = len(thresholded[0])
        g.info.height = len(thresholded)
        g.info.origin.position.x = yaml_data['origin'][0]
        g.info.origin.position.y = yaml_data['origin'][1]
        g.info.origin.position.z = yaml_data['origin'][2]
        g.info.origin.orientation.w = 1
        g.data = [item for sublist in thresholded for item in sublist]  # reshape
        self.map = g

    def map_callback(self, map):
        self.map = map

    def is_collision(self, pos):
        # pos is a (x,y,theta) tuple

        if self.map is None:
            rospy.logwarn("Plan requested before map was made available! Please try again later!")
            return GetPlanResponse()

        w = self.map.info.width
        h = self.map.info.height
        d = self.map.data
        res = self.map.info.resolution
        o = self.map.info.origin.position

        dbg("map metadata: {}".format(self.map.info))
        dbg("Origin of occupancy grid: ({},{})".format(o.x, o.y))

        x_mapspace = float(pos[0] - o.x) / res - GRID_OFFSET
        y_mapspace = float(pos[1] - o.y) / res - GRID_OFFSET
        data_index = int(round(y_mapspace)) * w + int(round(x_mapspace))
        return self.map.data[data_index] > 50

    def pub_map(self, timerEvent):
        if self.map is None:
            return
        self.map.header.stamp = rospy.Time.now()
        self.map_pub.publish(self.map)

    def start(self):
        rospy.Timer(rospy.Duration(1), self.pub_map)


def to_grayscale_array(img):
    img.load()
    if img.mode == 'RGBA':
        bg = Image.new('RGB', img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    img.convert(mode='RGB')
    img = np.array(img)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def main():
    rospy.init_node('map_service')
    ms = MapService(rospy.get_param('~map_directory'), None)
    ms.start()
    rospy.spin()
