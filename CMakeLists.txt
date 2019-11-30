cmake_minimum_required(VERSION 3.15)
project(camshift)

set(CMAKE_CXX_STANDARD 17)

add_executable(camshift main.cpp)

find_package(OpenCV REQUIRED)

target_link_libraries(camshift ${OpenCV_LIBS})