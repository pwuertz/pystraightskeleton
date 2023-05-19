cmake_minimum_required(VERSION 3.18)
project("pystraightskeleton" LANGUAGES CXX VERSION "1.0")

# Fetch conan.cmake
list(APPEND CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})
list(APPEND CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
if(NOT EXISTS "${CMAKE_BINARY_DIR}/conan.cmake")
  message(STATUS "Downloading conan.cmake from https://github.com/conan-io/cmake-conan")
  file(DOWNLOAD "https://raw.githubusercontent.com/conan-io/cmake-conan/0.18.1/conan.cmake"
       "${CMAKE_BINARY_DIR}/conan.cmake" TLS_VERIFY ON)
endif()
include(${CMAKE_BINARY_DIR}/conan.cmake)

# Get pybind11 and CGAL from conan
conan_cmake_configure(
  REQUIRES pybind11/[~2.10] cgal/[~5.5] boost/[~1.81]
  GENERATORS cmake_find_package
)
conan_cmake_autodetect(settings)
conan_cmake_install(
  PATH_OR_REFERENCE .
  BUILD missing
  REMOTE conancenter
  SETTINGS ${settings}
)

if(WIN32)
  # Prevent pybind11 from sharing resources with other, potentially ABI incompatible modules
  # https://github.com/pybind/pybind11/issues/2898
  add_definitions(-DPYBIND11_COMPILER_TYPE="_${PROJECT_NAME}_abi")
endif()

find_package(pybind11 REQUIRED)
find_package(CGAL REQUIRED)

pybind11_add_module(${PROJECT_NAME} "pystraightskeleton.cpp")
  set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 20
  CXX_STANDARD_REQUIRED ON
)
target_link_libraries(${PROJECT_NAME} PRIVATE CGAL::CGAL)
target_compile_definitions(${PROJECT_NAME} PRIVATE
  MODULE_NAME=${PROJECT_NAME}
  MODULE_VERSION=${PROJECT_VERSION}
)

message(${CGAL_INCLUDE_DIRS})

install(TARGETS ${PROJECT_NAME} DESTINATION ".")
