#!/usr/bin/env python
from sensor_msgs.msg import Imu
import tf
import rospy
from scipy.spatial import transform as stf

class Imu2Tf:
    def __init__(self):
        self.tfl = tf.TransformListener()
        self.tfb = tf.TransformBroadcaster()
        self.child_frame = rospy.get_param('~child_frame', 'base_link')
        self.fixed_frame = rospy.get_param('~fixed_frame', 'odom')
        self.sub_imu = rospy.Subscriber('imu/data', Imu, self.imu_callback)
        self.last_warn = 0.0

    def imu_callback(self, msg):
        R_world_to_imu = stf.Rotation.from_quat([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        try:
            (pos, orientation) = self.tfl.lookupTransform(msg.header.frame_id, self.child_frame, rospy.Time(0))
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            if rospy.Time.now().to_sec() - self.last_warn > 3.0:
                rospy.logwarn("Failed to find transformation between frames: {}".format(e))
                rospy.logwarn("What frame is the IMU publishing in?")
                self.last_warn = rospy.Time.now().to_sec()
            return
        R_imu_to_base = stf.Rotation.from_quat(orientation)

        q_out = (R_imu_to_base * R_world_to_imu).as_quat()

        self.tfb.sendTransform((0, 0, 0), q_out, rospy.Time.now(), self.child_frame, self.fixed_frame)


if __name__ == '__main__':
    rospy.init_node('imu2tf')
    i2t = Imu2Tf()
    rospy.spin()
