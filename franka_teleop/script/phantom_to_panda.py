import numpy as np
from scipy.spatial.transform import Rotation as R
import rospy
from omni_msgs.msg import OmniState, OmniButtonEvent, OmniFeedback
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion, WrenchStamped
from std_msgs.msg import Header
from franka_example_controllers.msg import ArmsTargetPose


def get_initial_pose():
    r = PoseStamped(header=Header(frame_id='rarm_link0'),
                    pose=Pose(position=Point(0.3, 0.0, 0.48),
                              orientation=Quaternion(1.0, 0.0, 0.0, 0.0)))
    l = PoseStamped(header=Header(frame_id='larm_link0'),
                    pose=Pose(position=Point(0.3, 0.0, 0.48),
                              orientation=Quaternion(1.0, 0, 0.0, 0.0)))
                              
    return r, l


class DualPhantomMaster(object):

    def __init__(self):        
        self.setup_ros()
        self.r_abs_pose, self.l_abs_pose = get_initial_pose()
        self.target_pose = ArmsTargetPose(right_target=self.r_abs_pose, left_target=self.l_abs_pose)
        self.send_rot = False
        self.pos_scale = 1.5
        self.force_scale = 0.2
        self.vel_scale = 0.001 * 0.001 * 1.5  # [1000Hz] * [m/mm] * [1.0 (scaler)]

    def setup_ros(self):
        rospy.init_node('DualPhantomMaster')
        rospy.Subscriber("/right_device/phantom/state", OmniState, self.right_device_cb)
        rospy.Subscriber("/left_device/phantom/state", OmniState, self.left_device_cb)

        rospy.Subscriber('/dual_panda/rarm_state_controller/F_ext', WrenchStamped, self.right_force_cb)
        rospy.Subscriber('/dual_panda/larm_state_controller/F_ext', WrenchStamped, self.left_force_cb)

        self.rarm_force_pub = rospy.Publisher('/right_device/phantom/force_feedback', OmniFeedback, queue_size=1)
        self.left_force_pub = rospy.Publisher('/left_device/phantom/force_feedback', OmniFeedback, queue_size=1)

        self.larm_target_pub = rospy.Publisher('/dual_panda/dual_arm_cartesian_pose_controller/larm_target_pose', PoseStamped, queue_size=1)
        self.rarm_target_pub = rospy.Publisher('/dual_panda/dual_arm_cartesian_pose_controller/rarm_target_pose', PoseStamped, queue_size=1)

        l_master_initial_q = R.from_quat([-0.5, -0.5, -0.5, 0.5])
        r_master_initial_q = R.from_quat([-0.5, -0.5, -0.5, 0.5])

        l_panda_initial_q = R.from_quat([1.0, 0, 0, 0])
        r_panda_initial_q = R.from_quat([1.0, 0, 0, 0])
        self.l_t = l_master_initial_q.inv() * l_master_initial_q
        self.r_t = r_master_initial_q.inv() * r_master_initial_q

        self.target_pub = rospy.Publisher('/dual_panda/dual_arm_cartesian_pose_controller/arms_target_pose', ArmsTargetPose, queue_size=1)
        self.right_zero_force = None
        self.left_zero_force = None
        rospy.loginfo("starting master node")

    def right_force_cb(self, msg):
        if self.right_zero_force is None:
            self.right_zero_force = np.array([msg.wrench.force.x,
                                              msg.wrench.force.y,
                                              msg.wrench.force.z])
        pub_msg = OmniFeedback()
        pub_msg.force.x = (msg.wrench.force.x - self.right_zero_force[0]) * self.force_scale
        pub_msg.force.y = (msg.wrench.force.y - self.right_zero_force[1]) * self.force_scale
        pub_msg.force.z = (msg.wrench.force.z - self.right_zero_force[2]) * self.force_scale
        self.rarm_force_pub.publish(pub_msg)

    def left_force_cb(self, msg):
        if self.left_zero_force is None:
            self.left_zero_force = np.array([msg.wrench.force.x,
                                             msg.wrench.force.y,
                                             msg.wrench.force.z])
        pub_msg = OmniFeedback()
        pub_msg.force.x = (msg.wrench.force.x - self.left_zero_force[0]) * self.force_scale
        pub_msg.force.y = (msg.wrench.force.y - self.left_zero_force[1]) * self.force_scale
        pub_msg.force.z = (msg.wrench.force.z - self.left_zero_force[2]) * self.force_scale
        self.left_force_pub.publish(pub_msg)
        
    def right_device_cb(self, msg):
        if msg.locked:  # not move when locked
            return
        self.r_abs_pose.pose.position.x += msg.velocity.y * self.vel_scale
        self.r_abs_pose.pose.position.y += -msg.velocity.x * self.vel_scale
        self.r_abs_pose.pose.position.z += msg.velocity.z * self.vel_scale

        if self.send_rot:
            cur_q = R.from_quat([msg.pose.orientation.x,
                                 msg.pose.orientation.y,
                                 msg.pose.orientation.z,
                                 msg.pose.orientation.w])
            tar_q = cur_q * self.r_t
            self.r_abs_pose.pose.orientation.x = tar_q.as_quat()[0]
            self.r_abs_pose.pose.orientation.y = tar_q.as_quat()[1]
            self.r_abs_pose.pose.orientation.z = tar_q.as_quat()[2]
            self.r_abs_pose.pose.orientation.w = tar_q.as_quat()[3]

        self.target_pose.right_target.pose = self.r_abs_pose.pose

    def left_device_cb(self, msg):
        if msg.locked:  # not move when locked
            return
        self.l_abs_pose.pose.position.x += msg.velocity.y * self.vel_scale
        self.l_abs_pose.pose.position.y += -msg.velocity.x * self.vel_scale
        self.l_abs_pose.pose.position.z += msg.velocity.z * self.vel_scale

        if self.send_rot:
            cur_q = R.from_quat([msg.pose.orientation.x,
                                 msg.pose.orientation.y,
                                 msg.pose.orientation.z,
                                 msg.pose.orientation.w])
            tar_q = cur_q * self.l_t
            self.l_abs_pose.pose.orientation.x = tar_q.as_quat()[0]
            self.l_abs_pose.pose.orientation.y = tar_q.as_quat()[1]
            self.l_abs_pose.pose.orientation.z = tar_q.as_quat()[2]
            self.l_abs_pose.pose.orientation.w = tar_q.as_quat()[3]

        self.target_pose.left_target.pose = self.l_abs_pose.pose

        self.rarm_target_pub.publish(self.r_abs_pose)
        self.larm_target_pub.publish(self.l_abs_pose)
        self.target_pub.publish(self.target_pose)

    def run(self):
        rospy.sleep(1.0)
        rospy.loginfo("Running...")
        rospy.spin()

    def __del__(self):
        rospy.loginfo("Exiting phantom master node")

def main():
    node = DualPhantomMaster()
    node.run()


if __name__ == '__main__':
    main()
