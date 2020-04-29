#!/usr/bin/env python

import rospy
from clifford_msgs.msg import CliffordCommand
from std_msgs.msg import Float64


class CliffordCmdMixing:
    def __init__(self):
        front_steering_controls = rospy.get_param("front_steering_controls")
        rear_steering_controls = rospy.get_param("rear_steering_controls")
        motor_speed_controls = rospy.get_param("motor_speed_controls")

        self.front_steering_pubs = []
        self.front_steering_scalars = []
        self.rear_steering_pubs = []
        self.rear_steering_scalars = []
        self.motor_speed_pubs = []

        for fsc in front_steering_controls:
            self.front_steering_scalars.append(fsc["scaling"])
            self.front_steering_pubs.append(rospy.Publisher(fsc["topic"], Float64, queue_size=10))
        for rsc in rear_steering_controls:
            self.rear_steering_scalars.append(rsc["scaling"])
            self.rear_steering_pubs.append(rospy.Publisher(rsc["topic"], Float64, queue_size=10))
        for msc in motor_speed_controls:
            self.motor_speed_pubs.append(rospy.Publisher(msc["topic"], Float64, queue_size=10))

        self.sub = rospy.Subscriber("cmd_clifford", CliffordCommand, self.cmd_robot)

    def cmd_robot(self, data):
        steer = data.steering
        throt = data.throttle
        drive_mode = data.steering_mode

        front_steering = 0
        back_steering = 0

        if drive_mode == CliffordCommand.DEFAULT:
            front_steering = steer
            back_steering = -steer
        elif data.steering_mode == CliffordCommand.FRONT_ONLY:
            front_steering = steer
        elif drive_mode == CliffordCommand.REAR_ONLY:
            back_steering = -steer
        elif drive_mode == CliffordCommand.STRAFE:
            front_steering = steer
            back_steering = steer

        for i, fsp in enumerate(self.front_steering_pubs):
            fsp.publish(front_steering * self.front_steering_scalars[i])
        for i, rsp in enumerate(self.rear_steering_pubs):
            rsp.publish(back_steering * self.rear_steering_scalars[i])
        for msp in self.motor_speed_pubs:
            msp.publish(throt)


if __name__ == '__main__':
    rospy.init_node('clifford_cmd_mixing')
    CCM = CliffordCmdMixing()
    rospy.spin()
