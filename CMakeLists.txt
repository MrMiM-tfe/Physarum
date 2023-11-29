set(CMAKE_C_COMPILER "C:\\msys64\\ucrt64\\bin\\g++.exe")
set(CMAKE_CXX_COMPILER "C:\\msys64\\ucrt64\\bin\\g++.exe")

cmake_minimum_required(VERSION 3.12)
project()

find_package(OpenCV REQUIRED)

add_executable(main main.cpp)

target_include_directories(main PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
