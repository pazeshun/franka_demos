(ros::load-ros-manifest "roseus")

(require :robot-interface "package://pr2eus/robot-interface.l")
(require :dual_franka "package://panda_eus/models/dual_panda.l")

(ros::roseus-add-msgs "franka_msgs")
(ros::roseus-add-msgs "franka_teleop")


(defun send-angle-vector (req)
  (print req)
  )
;;;
;;;
(ros::roseus "add_two_ints_server")
(ros::advertise-service "add_two_ints" roseus::AddTwoInts #'add-two-ints)
(do-until-key
 (ros::spin-once))
