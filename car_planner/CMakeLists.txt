cmake_minimum_required(VERSION 2.8.3)
project(car_planner)

add_compile_options(-Wall -Werror -std=c++14)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  message_generation
  sensor_msgs
  geometry_msgs
  message_filters
  robot_localization
  tf
  std_srvs
  )
find_package(yaml-cpp REQUIRED)

catkin_python_setup()

## Generate services in the 'srv' folder
#add_service_files(
#  FILES
#)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
)

catkin_package(CATKIN_DEPENDS
  std_msgs
  std_srvs
  roscpp
  rospy
  sensor_msgs
  geometry_msgs
  tf
  )

include_directories(
  src
  include
  ${catkin_INCLUDE_DIRS}
)

#catkin_install_python(PROGRAMS scripts/thruster_allocator
#                      DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY launch config include
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
  FILES_MATCHING PATTERN "*.h" PATTERN "*.hpp"
  PATTERN "*~" EXCLUDE)

# Macros for declaring ros nodes and libraries
include(cmake/ros-node.cmake)
include(cmake/cppcheck.cmake)
include(cmake/clang-format.cmake)
include(cmake/pylint.cmake)

# define nodes here
# eg: add_subdirectory(src/imu_calibrator)