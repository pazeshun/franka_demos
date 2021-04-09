#!/usr/bin/env roseus

(defun main ()
  (require :dual_panda-interface "package://panda_eus/euslisp/dual_panda-interface.l")

  (ros::roseus "dual_panda_realtime_ik")

  (dual_panda-init)

  (setq *robot* *dual_panda*)

  (send *robot* :angle-vector #f(45.0 -60.0 0.0 -120.0 0.0 60.0 90.0 -45.0 -60.0 0.0 -120.0 0.0 60.0 0.0))
  (when (send *ri* :check-error)
    (send *ri* :recover-error))

  (send *ri* :angle-vector (send *robot* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp :larm)
  (send *ri* :stop-grasp :rarm))

(main)
(eixt)
