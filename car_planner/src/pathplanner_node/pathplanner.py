import rospy
from nav_msgs.srv import GetPlan, GetPlanResponse
from nav_msgs.msg import Path, OccupancyGrid, MapMetaData
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from std_msgs.msg import Header
from scipy.spatial.transform import Rotation
from lib.FMT.test_fmt import get_path
import numpy as np
from visualization_msgs.msg import Marker


def tv(x):
    if isinstance(x, Point):
        return [x.x, x.y, x.z]
    if isinstance(x, Quaternion):
        return [x.x, x.y, x.z, x.w]


class PathPlanner:
    def __init__(self):
        self.map = None
        self.samples = None
        self.path = None
        self.map_sub = rospy.Subscriber('map', OccupancyGrid, self.map_callback)
        self.s = rospy.Service('plan_path', GetPlan, self.plan)
        self.path_pub = rospy.Publisher('path', Path, queue_size=10)
        self.samples_pub = rospy.Publisher('samples', Marker, queue_size=10)

    def plan(self, req):
        print('Processing path request!')

        if self.map is None:
            rospy.logwarn("No map published! Cannot plan without map metadata.")
            return GetPlanResponse()

        res = self.map.info.resolution  # meters per cell
        w = self.map.info.width
        h = self.map.info.height

        pos0_m = tv(req.start.pose.position)
        pos0_pixels = np.array(pos0_m) / res
        pos0 = [pos0_pixels[0] / h, pos0_pixels[1] / w]
        q0 = tv(req.start.pose.orientation)
        x0 = [pos0[0], pos0[1], Rotation.from_quat(q0).as_euler('ZYX')[0]]

        pos1_m = tv(req.goal.pose.position)
        pos1_pixels = np.array(pos1_m) / res
        pos1 = [pos1_pixels[0] / h, pos1_pixels[1] / w]
        q1 = tv(req.goal.pose.orientation)
        x1 = [pos1[0], pos1[1], Rotation.from_quat(q1).as_euler('ZYX')[0]]

        try:
            path = get_path(x0, x1)
        except Exception as e:
            rospy.logwarn("Failure to generate path! " + e.message)
            print("Failure to generate path!" + e.message)
            return GetPlanResponse()

        poses = []
        for x in list(path):
            p = PoseStamped()
            p.header.frame_id = 'world'
            p.pose.position = Point(x=x[0], y=x[1], z=0.1)
            q = Rotation.from_euler('ZYX', [x[2], 0, 0]).as_quat()
            p.pose.orientation = Quaternion(x=q[0], y=q[1], z=q[2], w=q[3])
            poses.append(p)

        path = Path()
        path.header.frame_id = 'world'
        path.header.stamp = rospy.Time.now()
        path.poses = poses

        self.path = path

        res = GetPlanResponse()
        res.plan = path
        print("Planning Successful!")

        return res

    def map_callback(self, map):
        self.map = map

    def publish_last(self, timer_event):
        if self.path is not None:
            self.path_pub.publish(self.path)
        if self.samples is not None:
            self.samples_pub.publish(self.samples)


def main():
    rospy.init_node('path_planner')
    pp = PathPlanner()
    rospy.Timer(rospy.Duration.from_sec(0.05), pp.publish_last)
    rospy.spin()
