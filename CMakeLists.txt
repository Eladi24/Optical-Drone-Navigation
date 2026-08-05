cmake_minimum_required(VERSION 3.16)
project(DroneNavigationHaifa CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Option to enable code coverage (default: OFF)
option(ENABLE_COVERAGE "Enable code coverage reporting" OFF)

# Configure coverage flags if enabled
if(ENABLE_COVERAGE)
    # Add coverage flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O0 --coverage -fprofile-arcs -ftest-coverage")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} --coverage -lgcov")
endif()

# OpenCV
find_package(OpenCV REQUIRED)

# libcurl (for HTTP calls to Google APIs)
find_package(CURL REQUIRED)

# pthread
find_package(Threads REQUIRED)

# Add include directories
include_directories(${OpenCV_INCLUDE_DIRS} ${CURL_INCLUDE_DIRS} .)
include_directories(${CMAKE_SOURCE_DIR}/include) 

# Build TemplateMatching as a library instead of an executable
add_library(DroneNavigation
    # Core simulation
    src/DroneSimulation.cpp
    
    # Algorithms
    src/FlightSimulation.cpp
    src/ORBFeatureEstimator.cpp
    src/SIFTFeatureEstimator.cpp
    src/HybridEstimator.cpp
    src/AKAZEFeatureEstimator.cpp
    
    # Visualization & utilities
    src/Visualization.cpp
    src/CoordinateUtils.cpp
    src/MapUtils.cpp
    src/KalmanFilter.cpp
    src/ParticleFilter.cpp
    
    # Video processing
    src/VideoProcessing.cpp
    src/GlobalLocator.cpp
    src/ThreadPool.cpp
    src/EdgeProcessor.cpp
    src/OpticalFlowTracker.cpp
    src/VideoPreprocessor.cpp
    src/GroundTruthAnnotator.cpp
    src/TelemetryImporter.cpp
    src/DatasetVideoAssembler.cpp
    src/SelfReferentialExperiment.cpp
)
target_link_libraries(DroneNavigation ${OpenCV_LIBS} ${CURL_LIBRARIES} Threads::Threads)

# ----- Google Maps + OpenCV sample -----
add_executable(Main src/Main.cpp)
target_link_libraries(Main ${OpenCV_LIBS} ${CURL_LIBRARIES} DroneNavigation)

# ----- Ground-truth annotation tool -----
# Interactive tool for manually geo-referencing landmark frames in the Haifa
# videos, used to build a small evaluation set (no telemetry/GPS exists to
# validate the navigation pipeline against otherwise).
add_executable(GroundTruth src/GroundTruthMain.cpp)
target_link_libraries(GroundTruth ${OpenCV_LIBS} ${CURL_LIBRARIES} DroneNavigation)

# Coverage target to generate HTML report with lcov
if(ENABLE_COVERAGE)
    # Check if lcov is installed
    find_program(LCOV lcov REQUIRED)
    find_program(GENHTML genhtml REQUIRED)
    
    add_custom_target(coverage
        # Capture coverage data
        COMMAND ${LCOV} --directory . --capture --output-file coverage.info
        # Filter out system headers and external libraries
        COMMAND ${LCOV} --remove coverage.info '/usr/*' '/lib/*' --output-file coverage.info
        # Generate HTML report
        COMMAND ${GENHTML} --output-directory coverage_report coverage.info
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        COMMENT "Generating code coverage report"
    )
endif()

# Build commands : cmake -S . -B build :configure
# cmake --build build -j :build
# cmake --build build --target clean :clean
# For coverage: cmake -S . -B build-coverage -DENABLE_COVERAGE=ON
# cmake --build build-coverage -j
# Run executable, then: cmake --build build-coverage --target coverage
# cmake --build build-coverage --target clean :clean coverage build
