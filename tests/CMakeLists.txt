include(FetchContent)
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    URL_HASH SHA256=7b42b4d6ed48810c5362c265a17faebe90dc2373c885e5216439d37927f02926
)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
set(BUILD_GMOCK OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# Shared helpers (procedural textures, temp dirs) live in test_helpers.hpp.
function(dronenav_add_test name label)
    add_executable(${name} ${ARGN})
    target_link_libraries(${name} PRIVATE DroneNavigation GTest::gtest_main)
    target_compile_options(${name} PRIVATE -Wall -Wextra)
    target_compile_definitions(${name} PRIVATE
        DRONENAV_FIXTURE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
        DRONENAV_SOURCE_DIR="${CMAKE_SOURCE_DIR}")
    if(DRONENAV_ENABLE_ONNX)
        target_link_libraries(${name} PRIVATE onnxruntime)
        set_target_properties(${name} PROPERTIES BUILD_RPATH ${onnxruntime_dist_SOURCE_DIR}/lib)
    endif()
    gtest_discover_tests(${name} PROPERTIES LABELS ${label} DISCOVERY_TIMEOUT 60)
endfunction()

dronenav_add_test(test_unit unit
    test_camera_model.cpp
    test_coordinate_utils.cpp
    test_reference_grid.cpp
    test_telemetry_importer.cpp
    test_vlad.cpp
    test_prewarp.cpp
    test_kalman.cpp
    test_thread_pool.cpp
    test_navigation_view.cpp)

dronenav_add_test(test_integration integration
    test_split_pipeline.cpp)

if(DRONENAV_ENABLE_ONNX)
    dronenav_add_test(test_local_only local_only
        test_matcher_cache.cpp)
endif()
