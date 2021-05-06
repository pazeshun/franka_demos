import numpy as np
import rospy
import rostopic
import yaml
import pickle
import franka_gripper.msg
from franka_gripper.msg import GraspEpsilon
from franka_teleop.srv import *
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
import message_filters
from franka_example_controllers.msg import ArmsTargetPose

from controller_manager_msgs.srv import *

import actionlib
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from trajectory_msgs.msg import JointTrajectoryPoint

JOINT_TRAJECTORY_JOINT_NAME = ["rarm_joint1",
                               "rarm_joint2",
                               "rarm_joint3",
                               "rarm_joint4",
                               "rarm_joint5",
                               "rarm_joint6",
                               "rarm_joint7",
                               "larm_joint1",
                               "larm_joint2",
                               "larm_joint3",
                               "larm_joint4",
                               "larm_joint5",
                               "larm_joint6",
                               "larm_joint7"]


class ImitationTaskExecuter(object):

    def __init__(self, initial_pose_file, topic_cfg_file, fps=50):
        rospy.init_node('ImitationTaskExecuter')
        self.initial_js = pickle.load(open(initial_pose_file, 'rb'))
        self.imitation_agent_srv = rospy.ServiceProxy('/command_imitation_agent', CommandImitationAgent)
        self.cfg = yaml.safe_load(open(topic_cfg_file, 'rb'))

        # Register message filter
        input_topic_list = []
        self.input_topitc_dict = {}
        self.input_data = {}
        for topic in self.cfg['input']:
            self.input_topitc_dict[topic] = rostopic.get_topic_type(topic)[0]
            input_topic_list.append(message_filters.Subscriber(topic,
                                                               rostopic.get_topic_class(topic)[0]))
            self.input_data[topic] = []
        # add image 
        input_topic_list.append(message_filters.Subscriber('/camera/rgb/image_raw', Image))
        self.fps = fps
        delay = 1.0/self.fps*0.5
        ts = message_filters.ApproximateTimeSynchronizer(input_topic_list, 10, delay)
        ts.registerCallback(self.input_topic_cb)

        # Register services
        self.switch_controller_srv = rospy.ServiceProxy('/dual_panda/controller_manager/switch_controller',
                                                        SwitchController)
        self.load_controller_srv = rospy.ServiceProxy('/dual_panda/controller_manager/load_controller',
                                                      LoadController)
        self.list_controller_srv = rospy.ServiceProxy('/dual_panda/controller_manager/list_controllers',
                                                      ListControllers)
        self.act_client = actionlib.SimpleActionClient('/dual_panda/dual_panda_effort_joint_trajectory_controller/follow_joint_trajectory',
                                                       FollowJointTrajectoryAction)
        # self.action_pubs = [rospy.Publisher('{}_test'.format(topic), rostopic.get_topic_class(topic)[0]) for topic in self.cfg['action']]
        self.l_test_pub = rospy.Publisher('/inference/right_target_inference', PoseStamped)
        self.r_test_pub = rospy.Publisher('/inference/left_target_inference', PoseStamped)
        self.action_pubs = [rospy.Publisher('/dual_panda/dual_arm_cartesian_pose_controller/arms_target_pose', ArmsTargetPose), 'GRASP_ARRAY']
        rospy.wait_for_service('/dual_panda/controller_manager/switch_controller')
        gripper_topics = ['/dual_panda/rarm/franka_gripper/grasp', '/dual_panda/larm/franka_gripper/grasp']
        self.gripper_clients = [actionlib.SimpleActionClient(gripper_topics[0], franka_gripper.msg.GraspAction),
                                actionlib.SimpleActionClient(gripper_topics[1], franka_gripper.msg.GraspAction)]
        ok = [c.wait_for_server() for c in self.gripper_clients]

        # Initialize member variabels
        self.input_vector = None
        self.input_images = None
        self.right_gripper_open = True
        self.left_gripper_open = True

    def input_topic_cb(self, *msgs):
        input_vector = []
        input_images = []
        for i, msg in enumerate(msgs):
            message_type = str(msg._type)
            if message_type == 'sensor_msgs/Image':
                input_images.append(msg)
                break
            topic = self.cfg['input'][i]
            raw_input_vector = []
            for attr in self.cfg['topics'][topic]:
                d = eval('msg.{}'.format(attr))
                if type(d) == tuple:
                    d = list(d)
                else:
                    d = [d]
                raw_input_vector += d
            input_vector += raw_input_vector
        self.input_vector = input_vector
        self.input_images = input_images

    def execute_action(self, action):
        counter = 0
        for pub, topic in zip(self.action_pubs, self.cfg['action']):
            # msg = rostopic.get_topic_class(topic)[0]()
            # TODO
            msg = ArmsTargetPose()
            if pub == 'GRASP_ARRAY':
                grippers_open = action[-2:]
                # TODO smater way
                # print(grippers_open)
                if grippers_open[0] > 0.5 and not self.right_gripper_open:
                    self.open_gripper('r')
                if grippers_open[1] > 0.5 and not self.left_gripper_open:
                    self.open_gripper('l')
                if grippers_open[0] <= 0.5 and self.right_gripper_open:
                    self.close_gripper('r')
                if grippers_open[0] <= 0.5 and self.left_gripper_open:
                    self.close_gripper('l')
                break   # TODO currently we assume grasps are located at the end of vector
            if str(msg._type) == 'franka_example_controllers/ArmsTargetPose':
                msg.right_target.header.frame_id = 'rarm_link0'
                msg.left_target.header.frame_id = 'larm_link0'
            for attr in self.cfg['topics'][topic]:
                exec('msg.{} = {}'.format(attr, str(action[counter])))
                counter += 1
            self.r_test_pub.publish(msg.right_target)
            self.l_test_pub.publish(msg.left_target)
            # pub.publish(msg)

    def open_gripper(self, arm):
        if arm == 'r':
            self.right_gripper_open = True
            ind = 0
        else:
            self.left_gripper_open = True
            ind = 1
        rospy.loginfo('Open {}arm'.format(arm))
        epsilon = GraspEpsilon(inner=0.01, outer=0.01)
        goal = franka_gripper.msg.GraspGoal(width=0.08, speed=1, force=10, epsilon=epsilon)
        self.gripper_clients[ind].send_goal(goal)
        
    def close_gripper(self, arm):
        if arm == 'r':
            self.right_gripper_open = False
            ind = 0
        else:
            self.left_gripper_open = False
            ind = 1
        rospy.loginfo('Clsoe {}arm'.format(arm))            
        # TODO, currently max force 140[N] is applied
        epsilon = GraspEpsilon(inner=0.01, outer=0.01)
        goal = franka_gripper.msg.GraspGoal(width=0, speed=1, force=140, epsilon=epsilon)
        self.gripper_clients[ind].send_goal(goal)
        

    def move_to_initial_pose(self, target_js=None):
        """Move arms to specified joint position.
        Input:
         target_js (sensor_msgs/JointState): target joint position3
        """
        if target_js is None:
            target_js = self.initial_js
        rospy.loginfo("Returning to initial position")
        # start follow_joint_trajectory controller
        controllers = self.list_controller_srv()
        if not True in ["dual_panda_effort_joint_trajectory_controller" in c.name for c in controllers.controller]:
            rospy.wait_for_service('/dual_panda/controller_manager/load_controller')
            self.load_controller_srv("dual_panda_effort_joint_trajectory_controller")
        rospy.wait_for_service('/dual_panda/controller_manager/switch_controller')
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
        self.act_client.wait_for_server()
        self.act_client.send_goal(traj_msg)
        self.act_client.wait_for_result()
        rospy.loginfo("Finish moving.")
        # unload follow_joint_trajectory controller and load dual_arm_cartesian_pose_controller
        self.switch_controller_srv(start_controllers=['dual_arm_cartesian_pose_controller'],
                                   stop_controllers=['dual_panda_effort_joint_trajectory_controller'],
                                   strictness=2,
                                   timeout=1.0)

    def start_inference(self):
        req = CommandImitationAgentRequest()
        counter = 0
        req = CommandImitationAgentRequest(command_type='reset')
        res = self.imitation_agent_srv(req)        
        while counter < 300:
            if self.input_vector is None or self.input_images is None:
                rospy.Rate(self.fps).sleep()                
                continue
            req = CommandImitationAgentRequest(command_type='act',
                                               input_vector=self.input_vector,
                                               input_images=self.input_images)
            res = self.imitation_agent_srv(req)
            self.execute_action(res.action)
            rospy.Rate(self.fps).sleep()
            counter += 1
        self.imitation_agent_srv

    def save_trajectory(self, path):
        NotImplementedError

    def spin(self):
        rospy.spin()


def main():
    initial_pose_file = '/home/ykawamura/jsk_imitation/data/yojo/2021-04-30-00-16-19/initial_joint_state.pkl'
    topic_cfg_file = '/home/ykawamura/ros/ws_franka/src/franka_demos/franka_teleop/config/rosbag_convert_config.yaml'
    node = ImitationTaskExecuter(initial_pose_file=initial_pose_file,
                                 topic_cfg_file=topic_cfg_file)
    rospy.loginfo('starting inference...')
    rospy.wait_for_service('/command_imitation_agent')
    node.move_to_initial_pose()
    rospy.sleep(1.0)
    node.start_inference()


if __name__ == '__main__':
    main()
