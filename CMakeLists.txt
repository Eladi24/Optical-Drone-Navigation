cmake_minimum_required(VERSION 3.16)
project(TemplateMatching CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# OpenCV
find_package(OpenCV REQUIRED)

# libcurl (for HTTP calls to Google APIs)
find_package(CURL REQUIRED)

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS} .)

# Build TemplateMatching as a library instead of an executable
add_library(TemplateMatching TemplateMatching.cpp)
target_link_libraries(TemplateMatching ${OpenCV_LIBS})

# ----- Google Maps + OpenCV sample -----
add_executable(Main Main.cpp)
target_link_libraries(Main ${OpenCV_LIBS} ${CURL_LIBRARIES} TemplateMatching)

# Build commands : cmake -S . -B build :configure
# cmake --build build -j :build
# cmake --build build --target clean :clean
