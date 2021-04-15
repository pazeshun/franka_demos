#!/usr/bin/env python
import os
from os import path
import numpy as np
import datetime
import rospy
import smach
import yaml
import pickle
from sensor_msgs.msg import Joy
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import JoyFeedback, JoyFeedbackArray, JointState
from franka_example_controllers.msg import ArmsTargetPose
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, WrenchStamped
from controller_manager_msgs.srv import *
from franka_teleop.srv import *

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint


INITIAL_X_RANGE = (0.0, 0.7)     # [m]
INITIAL_Y_RANGE = (-0.35, 0.35)  # [m]
INITIAL_Z_RANGE = (0.0, 0.7)     # [m]
JOINT_TRAJECTORY_JOINT_NAME = ["rarm_joint1", "rarm_joint2", "rarm_joint3", "rarm_joint4", "rarm_joint5", "rarm_joint6", "rarm_joint7", "larm_joint1", "larm_joint2", "larm_joint3", "larm_joint4", "larm_joint5", "larm_joint6", "larm_joint7"]


def get_initial_pose():
    r = PoseStamped(header=Header(frame_id='rarm_link0'),
                    pose=Pose(position=Point(0.3, 0.0, 0.48),
                              orientation=Quaternion(1.0, 0.0, 0.0, 0.0)))
    l = PoseStamped(header=Header(frame_id='larm_link0'),
                    pose=Pose(position=Point(0.3, 0.0, 0.48),
                              orientation=Quaternion(1.0, 0.0, 0.0, 0.0)))
    return r, l


def internal_division(move_range, cur):
    start = move_range[0]
    end = move_range[1]
    return (cur - start) / (end - start)


def inverse_internal_division(move_range, div):
    start = move_range[0]
    end = move_range[1]
    return (end - start) * div + start


def check_and_mkdir(dir_name):
    if not path.exists(dir_name):
        os.makedirs(dir_name)

        
def list_to_point(l):
    return Point(x=l[0], y=l[1], z=l[2])


def get_target_arms_pose_from_joy(msg):
    axes = np.array(msg.axes)
    r_targ, l_targ = get_initial_pose()    
    ranges = [INITIAL_X_RANGE, INITIAL_Y_RANGE, INITIAL_Z_RANGE]
    r_pos = [inverse_internal_division(r, p) for r,p in zip(ranges, axes[24:27])]
    l_pos = [inverse_internal_division(r, p) for r,p in zip(ranges, axes[27:30])]
    r_targ.pose.position = list_to_point(r_pos)
    l_targ.pose.position = list_to_point(l_pos)
    return ArmsTargetPose(right_target=r_targ, left_target=l_targ)

        
def get_joy_feedback_array(msg):
    if msg.header.frame_id == "rarm_link0":
        id_start = 0
    elif msg.header.frame_id == "larm_link0":
        id_start = 3
    else:
        raise AttributeError('Header frame of the current arm pose must be either [rarm|larm]_link0')
    x = JoyFeedback(type=0, id=id_start,   intensity=internal_division(INITIAL_X_RANGE, msg.pose.position.x))
    y = JoyFeedback(type=0, id=id_start+1, intensity=internal_division(INITIAL_Y_RANGE, msg.pose.position.y))
    z = JoyFeedback(type=0, id=id_start+2, intensity=internal_division(INITIAL_Z_RANGE, msg.pose.position.z))
    return JoyFeedbackArray([x, y, z])
        

class InitializeSetup(smach.State):

    def __init__(self, initial_pose_file=None):
        smach.State.__init__(self, outcomes=['done'])
        rospy.Subscriber('/midi_controller/joy', Joy, self.joy_cb)
        rospy.Subscriber('/dual_panda/dual_arm_cartesian_pose_controller/right_frame', PoseStamped, self.arm_cb)
        rospy.Subscriber('/dual_panda/dual_arm_cartesian_pose_controller/left_frame', PoseStamped, self.arm_cb)
        rospy.Subscriber('/dual_panda/joint_states', JointState, self.joint_states_cb)
        self.midi_pub = rospy.Publisher('/midi_controller/set_feedback', JoyFeedbackArray)
        self.target_pub = rospy.Publisher('/dual_panda/dual_arm_cartesian_pose_controller/arms_target_pose', ArmsTargetPose)
        self.user_control_slider = True
        self.initial_pose_file = initial_pose_file

    def execute(self, data):
        self.set_project_name()
        self.set_initial_pose()
        return 'done'
        
    def set_project_name(self):
        project_name = raw_input('Enter project name: ')
        if project_name == '':
            project_name = 'test'
        self.base_name = '/tmp/{}'.format(project_name)
        global dir_name        
        special_name = raw_input('Enter special name: ')
        time_stamp = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
        self.dir_name = path.join(self.base_name, special_name, time_stamp)
        rospy.loginfo('Rosbag will be saved in {}'.format(self.dir_name))
        dir_name = self.dir_name
        check_and_mkdir(self.dir_name)

    def set_initial_pose(self):
        if self.initial_pose_file is None:
            self.user_control_slider = False
            rospy.sleep(0.5)  # wait for slider to move
            self.user_control_slider = True        
            raw_input('Set initial pose by moving slider, press [Enter] to finalize.')
            global initial_js
            initial_js = self.cur_js
            self.dump_initial_pose_to_yaml(self.dir_name, initial_js)
        else:
            initial_js = pickle.load(open(initial_pose_file, 'rb'))
        
    def joint_states_cb(self, msg):
        self.cur_js = msg

    def arm_cb(self, msg):
        if not self.user_control_slider:
            self.midi_pub.publish(get_joy_feedback_array(msg))
        
    def joy_cb(self, msg):
        if self.user_control_slider:
            self.target_pub.publish(get_target_arms_pose_from_joy(msg))
    
    def dump_initial_pose_to_yaml(self, dir_name, js):
        with open(path.join(dir_name, 'initial_joint_state.pkl'), 'wb') as f:
            pickle.dump(js, f)
        obj = {'initial_pose': {'joint_names': list(js.name), 'pos': list(js.position)}}
        with open(path.join(dir_name, 'task_info.yaml'), 'wb') as f:
            yaml.dump(obj, f)
        

class TeachMotion(smach.State):
    def __init__(self):
        self.start_wait = 1.0
        smach.State.__init__(self, outcomes=['done', 'exit'])
        self.switch_controller_srv = rospy.ServiceProxy('/dual_panda/controller_manager/switch_controller',
                                                        SwitchController)
        self.load_controller_srv = rospy.ServiceProxy('/dual_panda/controller_manager/load_controller',
                                                      LoadController)
        self.list_controller_srv = rospy.ServiceProxy('/dual_panda/controller_manager/list_controllers',
                                                      ListControllers)
        self.act_client = actionlib.SimpleActionClient('/dual_panda/dual_panda_effort_joint_trajectory_controller/follow_joint_trajectory',
                                                       FollowJointTrajectoryAction)
        self.bilateral_control_srv=  rospy.ServiceProxy('/dual_panda/control_bilateral',
                                                        ControlBilateral)

    def execute(self, data):
        global initial_js        
        rospy.loginfo('Executing state TeachMotion')
        # move to initial pose 
        self.prepare_motion(initial_js)
        # start TouchUSB <-> Panda connection
        self.bilateral_control_srv(pose_connecting=True, force_connecting=True, wait=self.start_wait)
        self.start_rosbag_record()
        return 'done'

    def prepare_motion(self, target_js):
        rospy.loginfo("Returning to initial position")
        controllers = self.list_controller_srv.call()
        if len(controllers.controller) > 4:
            self.load_controller_srv('dual_panda_effort_joint_trajectory_controller')
        self.switch_controller_srv(start_controllers=['dual_panda_effort_joint_trajectory_controller'],
                                   stop_controllers=['dual_arm_cartesian_pose_controller'],
                                   strictness=2,
                                   timeout=1.0)
        rospy.loginfo("moving to initial position...")
        traj_msg = FollowJointTrajectoryGoal()
        traj_msg.trajectory.header.stamp = rospy.Time.now() + rospy.Duration(0.2)
        traj_msg.trajectory.joint_names = JOINT_TRAJECTORY_JOINT_NAME
        target_pos = target_js.position[0:7] + target_js.position[9:16]   # rarm + r_finger(skip) + larm
        traj_msg.trajectory.points.append(JointTrajectoryPoint(positions=target_pos,
                                                               time_from_start = rospy.Duration(2.0)))
        self.act_client.send_goal(traj_msg)
        self.act_client.wait_for_result()
        rospy.loginfo("Finish moving.")
        self.switch_controller_srv(start_controllers=['dual_arm_cartesian_pose_controller'],
                                   stop_controllers=['dual_panda_effort_joint_trajectory_controller'],
                                   strictness=2,
                                   timeout=1.0)

    def start_rosbag_record(self):
        rospy.loginfo("Start recoding rosbag...")
        pass


class LabelDemo(smach.State):
    def __init__(self):
        smach.State.__init__(self, outcomes=['done', 'exit'])

    def execute(self, data):
        rospy.loginfo('Executing state LabelDemo')
        rospy.sleep(2.0)
        return 'exit'


def main():
    rospy.init_node('smach_somple1')

    sm_top = smach.StateMachine(outcomes=['all_done'])
    with sm_top:
        smach.StateMachine.add('InitializeSetup', InitializeSetup(), transitions={'done':'TeachMotion'})
        smach.StateMachine.add('TeachMotion', TeachMotion(), transitions={'done':'LabelDemo', 'exit': 'all_done'})
        # smach.StateMachine.add('LabelDemo', TeachMotion(), transitions={'done':'TeachMotion', 'exit': 'all_done'})
        smach.StateMachine.add('LabelDemo', LabelDemo(), transitions={'done':'all_done', 'exit': 'all_done'})

    outcome = sm_top.execute()


if __name__ == '__main__':
    main()
