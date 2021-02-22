cmake_minimum_required(VERSION 2.8)
project( gms_match_demo)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)
# OpenCV 
find_package( OpenCV REQUIRED )

# 
include_directories( 
 include
${OpenCV_INCLUDE_DIRS} 
)


add_executable( gms_match_demo ./src/demo.cc )
target_link_libraries( gms_match_demo ${OpenCV_LIBS} Eigen3::Eigen)

add_executable( gms_in_lib ./src/gms_in_lib.cc )
target_link_libraries( gms_in_lib ${OpenCV_LIBS} Eigen3::Eigen)

