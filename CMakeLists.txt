cmake_minimum_required(VERSION 3.16)
project(control_design LANGUAGES CXX VERSION 1.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Dependencies. Eigen is header-only; a vendored copy under third_party/eigen
# is used when the system does not provide one.
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)

if(NOT Eigen3_FOUND)
    set(CONTROL_VENDORED_EIGEN "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
    if(EXISTS "${CONTROL_VENDORED_EIGEN}/Eigen/Core")
        message(STATUS "Using vendored Eigen at ${CONTROL_VENDORED_EIGEN}")
        add_library(control_eigen INTERFACE)
        # SYSTEM so Eigen's own -Wconversion/-Wsign-conversion noise does not
        # drown out this project's diagnostics.
        target_include_directories(control_eigen SYSTEM INTERFACE
            "${CONTROL_VENDORED_EIGEN}")
        add_library(Eigen3::Eigen ALIAS control_eigen)
    else()
        message(FATAL_ERROR
            "Eigen3 not found and no vendored copy at third_party/eigen. "
            "Install Eigen (e.g. 'brew install eigen') or unpack Eigen 3.4 into "
            "third_party/eigen.")
    endif()
else()
    message(STATUS "Using system Eigen ${Eigen3_VERSION}")
endif()

# ---------------------------------------------------------------------------
# Warnings
# ---------------------------------------------------------------------------
set(CONTROL_WARNINGS
    -Wall -Wextra -Wpedantic -Wshadow -Wnon-virtual-dtor -Wold-style-cast
    -Wcast-align -Wunused -Woverloaded-virtual -Wconversion -Wsign-conversion
    -Wdouble-promotion -Wformat=2)

# GCC reports false positives inside Eigen's vectorized kernels (a 16-byte SIMD
# load/store for an 8-byte fixed 1x1 Eigen matrix, diagnosed as an out-of-bounds
# access: arm_neon.h on aarch64, emmintrin.h on x86_64). Nothing here can fix
# that, only suppress it, so the suppression is narrow (GCC only, these three
# warnings) and applied after -Werror below, which would otherwise turn them
# into errors.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    list(APPEND CONTROL_WARNINGS
        -Wno-array-bounds -Wno-stringop-overread -Wno-stringop-overflow)
endif()

if(CONTROL_WERROR)
    list(APPEND CONTROL_WARNINGS -Werror)
endif()
option(CONTROL_WERROR "Treat warnings as errors" OFF)

# ---------------------------------------------------------------------------
# Library: controller design and analysis, for the host.
# ---------------------------------------------------------------------------
add_library(control
    src/analysis.cpp
    src/codegen.cpp
    src/design.cpp
    src/discretize.cpp
    src/lti.cpp
    src/mpc.cpp
    src/pid.cpp
    src/types.cpp)
add_library(control::control ALIAS control)

target_include_directories(control
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(control PUBLIC Eigen3::Eigen)
target_compile_options(control PRIVATE ${CONTROL_WARNINGS})

# ---------------------------------------------------------------------------
# Examples
# ---------------------------------------------------------------------------
add_executable(design_workflow_example examples/design_workflow_example.cpp)
target_link_libraries(design_workflow_example PRIVATE control::control)
target_compile_options(design_workflow_example PRIVATE ${CONTROL_WARNINGS})

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
enable_testing()

add_library(control_test_support STATIC tests/testing.cpp)
target_include_directories(control_test_support
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(control_test_support PUBLIC control::control)
target_compile_options(control_test_support PRIVATE ${CONTROL_WARNINGS})

# The generated controller is produced at build time, then compiled and compared
# against an independent implementation (the cross-check gate). Wiring it this way
# means the emitted code cannot rot: if the generator changes and the emitted
# header stops compiling or stops agreeing with the reference, the test fails.
set(GENERATED_CONTROLLER_DIR ${CMAKE_CURRENT_BINARY_DIR}/generated)
set(GENERATED_CONTROLLER ${GENERATED_CONTROLLER_DIR}/PositioningController.h)

add_executable(generate_controller examples/generate_controller.cpp)
target_link_libraries(generate_controller PRIVATE control::control)
target_include_directories(generate_controller PRIVATE examples)

add_custom_command(
    OUTPUT ${GENERATED_CONTROLLER}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${GENERATED_CONTROLLER_DIR}
    COMMAND generate_controller ${GENERATED_CONTROLLER}
    DEPENDS generate_controller
    COMMENT "Generating the deployable controller header")
add_custom_target(generated_controller_header DEPENDS ${GENERATED_CONTROLLER})

add_executable(generated_controller_test
    tests/generated_controller_test.cpp
    ${GENERATED_CONTROLLER})
add_dependencies(generated_controller_test generated_controller_header)
target_link_libraries(generated_controller_test PRIVATE control_test_support)
target_include_directories(generated_controller_test PRIVATE
    ${GENERATED_CONTROLLER_DIR} examples)

add_test(NAME generated_controller COMMAND generated_controller_test)

add_executable(control_tests
    tests/analysis_test.cpp
    tests/codegen_test.cpp
    tests/design_test.cpp
    tests/discretize_test.cpp
    tests/lti_test.cpp
    tests/mpc_test.cpp
    tests/pid_test.cpp
    tests/types_test.cpp)
target_link_libraries(control_tests PRIVATE control_test_support)
target_compile_options(control_tests PRIVATE ${CONTROL_WARNINGS})

add_test(NAME control_tests COMMAND control_tests)
