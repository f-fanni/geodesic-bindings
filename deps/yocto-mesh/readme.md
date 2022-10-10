# Yocto/Mesh: Extension of Yocto/GL for computational geometry

Yocto/Mesh is a collection of small C++17 libraries for building
computational geometry applications released under the MIT license.
Yocto/Mesh is an extension of Yocto/GL kept in a separate repository.

- `yocto_mesh/yocto_mesh.{h,cpp}`: computational geometry utilities for
  triangle meshes, mesh geodesic, mesh cutting

You can see Yocto/Mesh in action in the following applications written to
test the library:

- `apps/ymesh.cpp`: command-line mesh manipulation and rendering, and interactive viewing

For now, this extension keeps a whole copy of Yocto/GL inlined.
Eventually, we will remove the copy once it is clearer how to do it.
See [Yocto/GL](https://xelatihy.github.io/yocto-gl/) for more information on the project.
