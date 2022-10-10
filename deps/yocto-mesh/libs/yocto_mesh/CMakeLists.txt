add_library(yocto_mesh STATIC
  yocto_mesh.h yocto_mesh.cpp 
)

set_target_properties(yocto_mesh PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)
target_include_directories(yocto_mesh PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")

target_link_libraries(yocto_mesh PUBLIC yocto)

# warning flags
if(APPLE)
  target_compile_options(yocto_mesh PUBLIC -Wall -Wconversion -Wno-sign-conversion -Wno-implicit-float-conversion -Wno-unused-variable)
endif(APPLE)
if(MSVC)
  target_compile_options(yocto_mesh PUBLIC /D_CRT_SECURE_NO_WARNINGS)
  target_compile_options(yocto_mesh PUBLIC /EHsc)
endif(MSVC)
