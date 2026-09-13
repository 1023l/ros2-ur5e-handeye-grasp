cmake_minimum_required(VERSION 3.8)
project(arm_system)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

install(DIRECTORY
  config launch urdf rviz scripts models
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/wall_clock_publisher.py
  scripts/start_rviz.sh
  scripts/move_arm.py
  scripts/arm_perception.py
  scripts/perception_smoke.py
  scripts/arm_handeye.py
  scripts/approach_target.py
  scripts/arm_gripper.py
  scripts/pick_place.py
  scripts/calibrate_handeye.py
  scripts/setup_yolo_weights.py
  scripts/yolo_smoke.py
  scripts/arm_agent.py
  scripts/say.py
  scripts/eval_vision_grasp.py
  scripts/add_scene_obstacle.py
  scripts/eval_handeye_ba.py
  scripts/eval_handeye_ceres_style.py
  scripts/eval_planning_obstacle.py
  scripts/eval_planning_pareto.py
  scripts/run_sim_shift.py
  scripts/eval_stress.py
  scripts/eval_handeye_bakeoff.py
  scripts/eval_grasp_uncertainty.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
