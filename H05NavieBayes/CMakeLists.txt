
cmake_minimum_required(VERSION 2.8)
project(project_NaiveBayes)
set( CMAKE_CXX_STANDARD 11 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( THREADS_PREFER_PTHREAD_FLAG ON )

find_package( Threads REQUIRED )
find_package(glog REQUIRED)
find_package(Eigen3 REQUIRED)

set(GLOG_LIBRARIES "/usr/local/lib/libglog.so")

message(STATUS "GLOG_INCLUDE_DIRS: ${GLOG_INCLUDE_DIRS}")
message(STATUS "GLOG_LIBRARIES: ${GLOG_LIBRARIES}")
set(CMAKE_BUILD_TYPE "Debug")



set(CMAKE_INCLUDE_CURRENT_DIR ON)

aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} SRC_LIST)
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src/. SRC_LIST)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}
                    ${CMAKE_CURRENT_SOURCE_DIR}/include
                    ${EIGEN3_INCLUDE_DIR})
message(STATUS "GLOG_LIBRARIES: ${EIGEN3_INCLUDE_DIR}")

add_executable(${PROJECT_NAME} ${SRC_LIST})
target_include_directories(${PROJECT_NAME} PRIVATE ${GLOG_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE ${GLOG_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${EIGEN3_LIBRARIES})





