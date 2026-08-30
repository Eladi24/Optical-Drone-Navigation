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

# ONNX Runtime (STRATEGY.md Phase 2: learned retrieval/matching, frozen inference
# only). No apt package (libonnxruntime-dev doesn't exist on this project's
# reference system; libonnx-dev is the unrelated protobuf-schema-only ONNX
# format library) and no vcpkg/conan in this repo, so the prebuilt Linux x64
# release is fetched directly from Microsoft's GitHub releases and imported as
# a target by hand -- NOT via the release's own bundled onnxruntimeConfig.cmake
# (find_package(onnxruntime CONFIG)), which points INTERFACE_INCLUDE_DIRECTORIES
# at "include/onnxruntime" -- a directory that does not exist in this release
# layout (headers are directly under "include/"). Verified directly: config
# file present at lib/cmake/onnxruntime/onnxruntimeConfig.cmake in the 1.28.0
# release tarball references the wrong path, which would silently misconfigure
# the include dir and fail with "onnxruntime_cxx_api.h: No such file or
# directory" the first time it's actually built against, not at configure time.
option(DRONENAV_ENABLE_ONNX "Enable ONNX Runtime-backed learned retrieval/matching stages" ON)

if(DRONENAV_ENABLE_ONNX)
    include(FetchContent)
    set(ONNXRUNTIME_VERSION "1.28.0")
    FetchContent_Declare(
        onnxruntime_dist
        URL https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-${ONNXRUNTIME_VERSION}.tgz
        URL_HASH SHA256=a3e1b79d7bb1bf09696ce675f49e4064e6c81f6202b8225624fff0e93f8d6407
    )
    FetchContent_MakeAvailable(onnxruntime_dist)

    add_library(onnxruntime SHARED IMPORTED GLOBAL)
    set_target_properties(onnxruntime PROPERTIES
        IMPORTED_LOCATION             ${onnxruntime_dist_SOURCE_DIR}/lib/libonnxruntime.so
        INTERFACE_INCLUDE_DIRECTORIES ${onnxruntime_dist_SOURCE_DIR}/include
    )
endif()

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
    src/HistogramRetrieval.cpp
    src/OrbRansacMatching.cpp
    src/SplitPipelineEstimator.cpp

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
if(DRONENAV_ENABLE_ONNX)
    # STRATEGY.md Phase 2 learned retrieval/matching (see include/OnnxDeitRetrieval.hpp,
    # include/XFeatMatching.hpp, include/DiskLightGlueMatching.hpp, include/DinoDenseMatching.hpp).
    target_sources(DroneNavigation PRIVATE src/OnnxDeitRetrieval.cpp src/XFeatMatching.cpp
                                            src/DiskLightGlueMatching.cpp src/DinoDenseMatching.cpp)
    target_link_libraries(DroneNavigation onnxruntime)
endif()

# ----- Google Maps + OpenCV sample -----
add_executable(Main src/Main.cpp)
target_link_libraries(Main ${OpenCV_LIBS} ${CURL_LIBRARIES} DroneNavigation)

# ----- Ground-truth annotation tool -----
# Interactive tool for manually geo-referencing landmark frames in the Haifa
# videos, used to build a small evaluation set (no telemetry/GPS exists to
# validate the navigation pipeline against otherwise).
add_executable(GroundTruth src/GroundTruthMain.cpp)
target_link_libraries(GroundTruth ${OpenCV_LIBS} ${CURL_LIBRARIES} DroneNavigation)

# The fetched onnxruntime shared library lives under build/_deps/, not a system
# library path -- without an explicit RPATH, Main/GroundTruth link fine but fail
# at RUN time with "error while loading shared libraries: libonnxruntime.so.1:
# cannot open shared object file". Verified this is genuinely needed (not
# hypothetical) by checking libonnxruntime.so.1.28.0 is not installed anywhere
# under the usual system library paths.
if(DRONENAV_ENABLE_ONNX)
    set_target_properties(Main GroundTruth PROPERTIES
        BUILD_RPATH ${onnxruntime_dist_SOURCE_DIR}/lib)
endif()

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
