#!/usr/bin/env roseus
(setq init-1 #f(45.0 -60.0 0.0 -120.0 0.0 60.0 90.0 -45.0 -60.0 0.0 -120.0 0.0 60.0 0.0)))
(setq init-2 #f(79.9176 -38.9664 -64.8957 -138.625 -36.8676 112.484 82.1401 -75.9776 -46.0815 61.3643 -145.73 41.9323 110.964 9.04242))
(defun main ()
  (require :dual_panda-interface "package://panda_eus/euslisp/dual_panda-interface.l")

  (ros::roseus "dual_panda_realtime_ik")

  (dual_panda-init)

  (setq *robot* *dual_panda*)

  (send *robot* :angle-vector init-2)
  (when (send *ri* :check-error)
    (send *ri* :recover-error))
  (send *ri* :angle-vector (send *robot* :angle-vector) 3000)
  (send *ri* :wait-interpolation)
  (send *ri* :stop-grasp :larm)
  (send *ri* :stop-grasp :rarm))

(main)
(eixt)
