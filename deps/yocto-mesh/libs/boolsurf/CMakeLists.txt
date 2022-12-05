add_subdirectory(ext/CDT/CDT)

add_library(boolsurf
  boolsurf_utils.h
  boolsurf.h boolsurf.cpp
  #boolsurf_io.h boolsurf_io.cpp
)

set_target_properties(boolsurf PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(boolsurf PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(boolsurf PUBLIC yocto_mesh)

#find_package(cinolib REQUIRED)
#target_link_libraries(boolsurf PUBLIC cinolib)

if(WIN32)
  target_compile_definitions(boolsurf PUBLIC "NOMINMAX")
  target_compile_definitions(boolsurf PUBLIC "WIN32_LEAN_AND_MEAN")
endif()

# warning flags
if(APPLE)
  target_compile_options(yocto PUBLIC -Wall -Wconversion -Wno-sign-conversion -Wno-implicit-float-conversion)
endif(APPLE)
if(MSVC)
  # target_compile_options(yocto PUBLIC /D_CRT_SECURE_NO_WARNINGS /wd4018 /wd4244 /wd4305 /wd4800 /wd4267)
  target_compile_options(yocto PUBLIC /D_CRT_SECURE_NO_WARNINGS)
  target_compile_options(yocto PUBLIC /EHsc)
endif(MSVC)