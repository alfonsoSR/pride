cmake_minimum_required(VERSION 3.17.2...3.29)
project(${SKBUILD_PROJECT_NAME} LANGUAGES C Fortran)

file(GLOB files "/usr/bin/*")
foreach (file ${files})
    message(STATUS "File: ${file}")
endforeach()

find_package(
  Python
  COMPONENTS Interpreter Development.Module NumPy
  REQUIRED
)

# Alias for source directiories
set(IERS_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/iers")
set(VIENNA_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/vienna")
set(PRIDE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/src/pride")

# Import add_extension macro
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FortranExtension)

fortran_extension(${IERS_SOURCE_DIR}/dehanttideinel external/iers)
fortran_extension(${IERS_SOURCE_DIR}/eops external/iers)
fortran_extension(${IERS_SOURCE_DIR}/hardisp external/iers)
fortran_extension(${VIENNA_SOURCE_DIR}/vmf3 external/vienna)

install(
    DIRECTORY ${PRIDE_SOURCE_DIR}
    DESTINATION .
)
