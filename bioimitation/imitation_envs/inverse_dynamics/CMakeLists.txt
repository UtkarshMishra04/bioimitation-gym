cmake_minimum_required(VERSION 3.0)

# project
project(inverse_dynamics VERSION 1.0.0)

# for emacs completion
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# for grouping into folders
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

# (backward compatibility with older compilers); c++11, -g option is
# used to export debug symbols for gdb
if(${CMAKE_CXX_COMPILER_ID} MATCHES GNU OR
   ${CMAKE_CXX_COMPILER_ID} MATCHES Clang)
  # Using C++11 on OSX requires using libc++ instead of libstd++.
  # libc++ is an implementation of the C++ standard library for OSX.
  if(APPLE)
    if(XCODE)
      set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LANGUAGE_STANDARD "c++11")
      set(CMAKE_XCODE_ATTRIBUTE_CLANG_CXX_LIBRARY "libc++")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++")
    endif()
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -g -Wno-deprecated -w")
    # -fno-var-tracking-assignments
  endif()
endif()

# configure for new version of OpenSim
find_package(OpenSim REQUIRED)
include_directories(${OpenSim_INCLUDE_DIRS})
link_directories(${OpenSim_LIB_DIR})

# configure Python and Boost
find_package(PythonLibs ${PY_VERSION} REQUIRED)
find_package(Boost REQUIRED COMPONENTS python)
include_directories(${PYTHON_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

# create the lib
set(target inverse_dynamics)
add_library(${target} SHARED inverse_dynamics.cpp)
target_link_libraries(${target} ${Boost_LIBRARIES} ${OpenSim_LIBRARIES})
# suppress prefix "lib" because Python does not allow this prefix
set_target_properties(${target} PROPERTIES PREFIX "")

set(ADDITIONAL_FILES
  "__init__.py"
  "test_inverse_dynamics.py"
  "opensim_utils.py"
  "model_scaled.osim"
  "task_Kinematics_q.sto"
  "task_Kinematics_u.sto"
 #  "task_Kinematics_dudt.sto"
)

foreach(dataFile ${ADDITIONAL_FILES})
  file(COPY "${dataFile}" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}")
endforeach()
