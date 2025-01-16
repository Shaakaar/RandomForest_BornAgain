cmake_minimum_required(VERSION 3.16.1)

project(MurTree VERSION 1.0.0)

# ------------------------------------------------------------------------------
# 1. Basic build-type setup
# ------------------------------------------------------------------------------
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  message(FATAL_ERROR
    "Please create a separate build directory instead of building in-source. "
    "You may need to remove CMakeCache.txt."
  )
endif()

  find_package(Python3 COMPONENTS Development REQUIRED)
  include_directories("C:/Users/shaka/AppData/Local/Programs/Python/Python313/include")
  link_directories("C:/Users/shaka/AppData/Local/Programs/Python/Python313/libs")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

string(TOLOWER "${CMAKE_BUILD_TYPE}" cmake_build_type_tolower)
if(NOT cmake_build_type_tolower STREQUAL "debug"
   AND NOT cmake_build_type_tolower STREQUAL "release")
  message(FATAL_ERROR
    "Unknown build type \"${CMAKE_BUILD_TYPE}\". Allowed values: Debug or Release.")
endif()

set(CMAKE_CXX_STANDARD 14)

# ------------------------------------------------------------------------------
# 2. Compiler flags
# ------------------------------------------------------------------------------
if(MSVC)
  # Force always compile with warning level 3
  if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
    string(REGEX REPLACE "/W[0-4]" "/W3" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W3")
  endif()
elseif(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -O3 -std=c++14 -DNDEBUG -Wall")
endif()

# ------------------------------------------------------------------------------
# 3. Source files for MurTree
# ------------------------------------------------------------------------------
include_directories(
  "code/MurTee/Data Structures/"
  code/MurTree/Engine/
  code/MurTree/Utilities/
)

set(DATA_STRUCTURES_SRC
  "code/MurTree/Data Structures/binary_data.cpp"
  "code/MurTree/Data Structures/branch.cpp"
  "code/MurTree/Data Structures/decision_tree.cpp"
  "code/MurTree/Data Structures/feature_vector_binary.cpp"
  "code/MurTree/Data Structures/internal_node_description.cpp"
  "code/MurTree/Data Structures/key_value_heap.cpp"
  "code/MurTree/Data Structures/symmetric_matrix_counter.cpp"
  "code/MurTree/Data Structures/symmetric_matrix_positive_negative_counter_2d.cpp"
)

set(ENGINE_SRC
  code/MurTree/Engine/binary_data_difference_computer.cpp
  code/MurTree/Engine/branch_cache.cpp
  code/MurTree/Engine/cache_closure.cpp
  code/MurTree/Engine/dataset_cache.cpp
  code/MurTree/Engine/feature_selector_abstract.cpp
  code/MurTree/Engine/hyper_parameter_tuner.cpp
  code/MurTree/Engine/similarity_lower_bound_computer.cpp
  code/MurTree/Engine/solver.cpp
  code/MurTree/Engine/specialised_binary_classification_decision_tree_solver.cpp
  code/MurTree/Engine/specialised_branch_misclassification_computer.cpp
  code/MurTree/Engine/specialised_general_branch_misclassification_computer.cpp
  code/MurTree/Engine/specialised_general_classification_decision_tree_solver.cpp
)

set(UTILITIES_SRC
  code/MurTree/Utilities/file_reader.cpp
  code/MurTree/Utilities/parameter_handler.cpp
)

# ------------------------------------------------------------------------------
# 4. Build options: 1 standalone, 2 shared lib, 3 python module
# ------------------------------------------------------------------------------
option(BUILD_STANDALONE_APPLICATION "Build MurTree as a standalone application" OFF)
option(BUILD_SHARED_LIBRARY "Build MurTree as a shared library" OFF)
option(BUILD_PYTHON_MODULE "Build MurTree as a Python moduke" ON)

if(BUILD_STANDALONE_APPLICATION AND NOT BUILD_SHARED_LIBRARY AND NOT BUILD_PYTHON_MODULE)
  # 4A. Standalone
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

  add_executable(murtree
    main.cpp
    ${DATA_STRUCTURES_SRC}
    ${ENGINE_SRC}
    ${UTILITIES_SRC}
  )

elseif(BUILD_SHARED_LIBRARY AND NOT BUILD_STANDALONE_APPLICATION AND NOT BUILD_PYTHON_MODULE)
  # 4B. Shared library
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
  add_compile_definitions(BUILD_LIBRARY)
  add_library(murtree SHARED
    main.cpp
    ${DATA_STRUCTURES_SRC}
    ${ENGINE_SRC}
    ${UTILITIES_SRC}
  )

elseif(BUILD_PYTHON_MODULE AND NOT BUILD_STANDALONE_APPLICATION AND NOT BUILD_SHARED_LIBRARY)
  # Build as a Python module
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
  add_compile_definitions(BUILD_PYTHON_MODULE)

  # Because the code has #include "include/pybind11/pybind11.h"
  # we must point to the folder that has 'include/pybind11/pybind11.h' below it
  # i.e. <pybind11_top>/include/pybind11/pybind11.h
  # AND we also need a second path so that <pybind11/attr.h> is found.

  set(PYBIND11_DIR "C:/Users/shaka/OneDrive/Email attachments/Documenten/Data Science Project/MurTree-Master/venv/Lib/site-packages/pybind11")

  # 1 This path => so the code's #include "include/pybind11/pybind11.h" resolves
  #    to <PYBIND11_DIR>/include/pybind11/pybind11.h
  # 2 This path => so that <pybind11/attr.h> is found in <PYBIND11_DIR>/include
  include_directories(
    "${PYBIND11_DIR}"          # needed for "include/pybind11/pybind11.h"
    "${PYBIND11_DIR}/include"  # needed for <pybind11/attr.h>
  )

  add_library(murtree_python_module MODULE
    main.cpp
    ${DATA_STRUCTURES_SRC}
    ${ENGINE_SRC}
    ${UTILITIES_SRC}
  )

 set_target_properties(murtree_python_module PROPERTIES
    PREFIX ""  # Avoids adding "lib" prefix on Windows
    SUFFIX ".pyd"  # Ensures the output has the correct extension
    OUTPUT_NAME "murtree_python_module"
)


else()
  message(FATAL_ERROR
    "Please select exactly ONE build: BUILD_STANDALONE_APPLICATION, BUILD_SHARED_LIBRARY, or BUILD_PYTHON_MODULE."
  )
endif()
