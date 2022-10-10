//
// # Yocto/Mesh: Tiny Library for mesh operations for computational geometry
//
// Yocto/Mesh is a collection of computational geometry routines on triangle
// meshes.
//

//
// LICENSE:
//
// Copyright (c) 2016 -- 2021 Fabio Pellacini
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//

#ifndef _YOCTO_MESH_H_
#define _YOCTO_MESH_H_

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <yocto/yocto_math.h>

// -----------------------------------------------------------------------------
// USING DIRECTIVES
// -----------------------------------------------------------------------------
namespace yocto {

// using directives
using std::array;
using std::pair;
using std::string;
using std::unordered_map;
using std::vector;

}  // namespace yocto

// -----------------------------------------------------------------------------
// PROCEDURAL MODELING
// -----------------------------------------------------------------------------
namespace yocto {

// Extract isoline from surface scalar field.
void meandering_triangles(const vector<float>& field, float isoline,
    int selected_tag, int t0, int t1, vector<vec3i>& triangles,
    vector<int>& tags, vector<vec3f>& positions, vector<vec3f>& normals);

}  // namespace yocto

// -----------------------------------------------------------------------------
// ADJACENCIES
// -----------------------------------------------------------------------------
namespace yocto {

// Returns the list of triangles incident at each vertex in CCW order.
// Note: this works only if the mesh does not have a boundary.
vector<vector<int>> vertex_to_triangles(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies);

// Face adjacent to t and opposite to vertex vid
int opposite_face(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, int t, int vid);

// Finds the opposite vertex of an edge
int opposite_vertex(const vec3i& triangle, const vec2i& edge);
int opposite_vertex(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, int face, int k);
int common_vertex(const vector<vec3i>& triangles, int pid0, int pid1);

// Finds common edge between triangles
vec2i common_edge(const vec3i& triangle0, const vec3i& triangle1);
vec2i opposite_edge(const vec3i& t, int vid);
vec2i common_edge(const vector<vec3i>& triangles, int pid0, int pid1);

// Triangle fan starting from a face and going towards the k-th neighbor face.
vector<int> triangle_fan(
    const vector<vec3i>& adjacencies, int face, int k, bool clockwise = false);

}  // namespace yocto

// -----------------------------------------------------------------------------
// GEODESIC COMPUTATION
// -----------------------------------------------------------------------------
namespace yocto {

// Data structure used for geodesic computation
struct geodesic_solver {
  static const int min_arcs = 12;
  struct graph_edge {
    int   node   = -1;
    float length = flt_max;
  };
  vector<vector<graph_edge>> graph = {};
};

// Construct a graph to compute geodesic distances
geodesic_solver make_geodesic_solver(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, const vector<vec3f>& positions);

// Construct a graph to compute geodesic distances with arcs arranged in
// counterclockwise order by using the vertex-to-face adjacencies
geodesic_solver make_geodesic_solver(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const vector<vector<int>>& vertex_to_faces);

// Compute geodesic distances
vector<float> compute_geodesic_distances(const geodesic_solver& solver,
    const vector<int>& sources, float max_distance = flt_max);

// Compute geodesic distances
void update_geodesic_distances(vector<float>& distances,
    const geodesic_solver& solver, const vector<int>& sources,
    float max_distance = flt_max);

// Compute all shortest paths from source vertices to any other vertex.
// Paths are implicitly represented: each node is assigned its previous node
// in the path. Graph search early exits when reching end_vertex.
vector<int> compute_geodesic_parents(const geodesic_solver& solver,
    const vector<int>& sources, int end_vertex = -1);

// Sample vertices with a Poisson distribution using geodesic distances.
// Sampling strategy is farthest point sampling (FPS): at every step
// take the farthers point from current sampled set until done.
vector<int> sample_vertices_poisson(
    const geodesic_solver& solver, int num_samples);

// Compute the distance field needed to compute a voronoi diagram
vector<vector<float>> compute_voronoi_fields(
    const geodesic_solver& solver, const vector<int>& generators);

// Convert distances to colors
vector<vec3f> colors_from_field(const vector<float>& field, float scale = 1,
    const vec3f& c0 = {1, 1, 1}, const vec3f& c1 = {1, 0.1f, 0.1f});

struct mesh_point {
  int   face = -1;
  vec2f uv   = {0, 0};
};

// Compute position and normal of a point on the mesh via linear interpolation
vec3f eval_position(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const mesh_point& sample);
vec3f eval_normal(const vector<vec3i>& triangles, const vector<vec3f>& normals,
    const mesh_point& point);
}

// -----------------------------------------------------------------------------
// STRIPS
// -----------------------------------------------------------------------------

namespace yocto {
// Dual geodesic graph
struct dual_geodesic_solver {
  struct edge {
    int   node   = -1;
    float length = flt_max;
  };
  vector<array<edge, 3>> graph     = {};
  vector<vec3f>          centroids = {};
};

// Construct a graph to compute geodesic distances
dual_geodesic_solver make_dual_geodesic_solver(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies);

struct strip_arena {
  vector<int> parents;
  vector<float> field;
  vector<bool> in_queue;
  vector<int> visited;

  strip_arena(size_t size);
  void cleanup();
};

inline auto null_arena = strip_arena(0);

vector<int> compute_short_strip(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const mesh_point& start, const mesh_point& end, strip_arena& arena = null_arena);

// Compute visualizations for the shortest path connecting a set of points.
vector<vec3f> visualize_shortest_path(const dual_geodesic_solver& graph,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& start,
    const mesh_point& end, strip_arena& arena = null_arena);
vector<vec3f> visualize_shortest_path(const dual_geodesic_solver& graph,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<mesh_point>& points, strip_arena& arena = null_arena);

}  // namespace yocto


// -----------------------------------------------------------------------------
// GEODESIC PATHS
// -----------------------------------------------------------------------------

namespace yocto {
struct geodesic_path {
  // surface data
  mesh_point    start = {};
  mesh_point    end   = {};
  vector<int>   strip = {};
  vector<float> lerps = {};
};

// Compute the shortest path connecting two surface points.
geodesic_path compute_shortest_path(const dual_geodesic_solver& graph,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const mesh_point& start,
    const mesh_point& end, strip_arena& arena = null_arena);

// Compute the shortest path connecting a set of points.
// vector<mesh_point> compute_shortest_path(const dual_geodesic_solver& graph,
//     const vector<vec3i>& triangles, const vector<vec3f>& positions,
//     const vector<vec3i>& adjacencies, const vector<mesh_point>& points);

// compute the straightest path given a surface point and tangent direction
geodesic_path compute_straightest_path(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const mesh_point& start, const vec2f& direction, float path_length);
} 

// -----------------------------------------------------------------------------
// PARALLEL TRANSPORT AND ANGLES
// -----------------------------------------------------------------------------
namespace yocto {

// TODO(fabio): implement wrapper
// compute the 2d rotation in tangent space that tansport directions from
// the staring point of the path to its ending point.
mat2f parallel_transport_rotation(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies);

// Compute angles in tangent space and total angles of every vertex
vector<vector<float>> compute_angles(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const vector<vector<int>>& v2t, vector<float>& total_angles,
    bool with_opposite);

}  // namespace yocto

// -----------------------------------------------------------------------------
// GEODESIC PATH
// -----------------------------------------------------------------------------
namespace yocto {

using mesh_path = vector<mesh_point>;

mesh_path convert_mesh_path(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, const vector<int>& strip,
    const vector<float>& lerps, const mesh_point& start, const mesh_point& end);

inline mesh_path convert_mesh_path(const vector<vec3i>& triangles,
    const vector<vec3i>& adjacencies, const geodesic_path& path) {
  return convert_mesh_path(
      triangles, adjacencies, path.strip, path.lerps, path.start, path.end);
}

// compute the shortest path connecting two surface points
// initial guess of the connecting strip must be given
geodesic_path shortest_path(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const mesh_point& start, const mesh_point& end, const vector<int>& strip);

// compute the 2d rotation in tangent space that tansport directions from
// the staring point of the path to its ending point.
mat2f parallel_transport_rotation(const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const geodesic_path& path);

vec2f tangent_path_direction(const geodesic_path& path, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies, bool start = true);

vector<vec3f> path_positions(const geodesic_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies);
vector<vec3f> path_positions(const mesh_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies);

vector<float> path_parameters(const geodesic_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies);
vector<float> path_parameters(const mesh_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies);

float path_length(const geodesic_path& path, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies);
float path_length(const mesh_path& path, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies);

vector<float> path_parameters(const vector<vec3f>& positions);
float         path_length(const vector<vec3f>& positions);

mesh_point eval_path_point(const geodesic_path& path,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, float t);

}  // namespace yocto


// -----------------------------------------------------------------------------
// BEZIER
// -----------------------------------------------------------------------------
namespace yocto {
// compute a bezier on the surface
vector<mesh_point> compute_bezier_uniform(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<mesh_point>& control_points,
    int subdivision = 4, strip_arena& arena = null_arena);
// compute a bezier on the surface
vector<mesh_point> compute_bezier_uniform(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies,
    const array<mesh_point, 4>& control_points, int subdivision = 4, strip_arena& arena = null_arena);

vector<mesh_point> compute_bezier_adaptive(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const array<mesh_point, 4>& control_points, int max_depth, float min_curve_size, float precision,
    strip_arena& arena);

// evaluates a point a bezier by subdivision
mesh_point eval_bezier_point(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const array<mesh_point, 4>& segment,
    float t, float precision = 0.1f, strip_arena& arena = null_arena);

// evaluates a point a bezier by subdivision
array<array<mesh_point, 4>, 2> insert_bezier_point(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const array<mesh_point, 4>& segment, float t,
    float precision = 0.1f, strip_arena& arena = null_arena);

} // namespace yocto



// -----------------------------------------------------------------------------
// LANE-RIESENFELD 
// -----------------------------------------------------------------------------

namespace yocto {

enum struct spline_algorithm {
  de_casteljau_uniform = 0,
  de_casteljau_adaptive,
  lane_riesenfeld_uniform,
  lane_riesenfeld_adaptive
};
const auto spline_algorithm_names = vector<string>{
    "dc-uniform", "dc-adaptive", "lr-uniform", "lr-adaptive"};

struct spline_params {
  spline_algorithm algorithm      = spline_algorithm::de_casteljau_uniform;
  int              subdivisions   = 4;
  float            precision      = 0.1f;
  float            min_curve_size = 0.001f;
  int              max_depth      = 10;
};

// compute a bezier on the surface
vector<mesh_point> compute_bezier_path(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<mesh_point>& control_points,
    const spline_params& params, strip_arena& arena = null_arena);
vector<mesh_point> compute_bezier_path(const dual_geodesic_solver& solver,
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>&        adjacencies,
    const array<mesh_point, 4>& control_points, const spline_params& params, strip_arena& arena = null_arena);


vector<mesh_point> lane_riesenfeld_uniform(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const array<mesh_point, 4>& control_points, int num_subdivisions, strip_arena& arena);
vector<mesh_point> lane_riesenfeld_adaptive(
    const dual_geodesic_solver& solver, const vector<vec3i>& triangles,
    const vector<vec3f>& positions, const vector<vec3i>& adjacencies,
    const array<mesh_point, 4>& polygon, const spline_params& params, strip_arena& arena);

}  // namespace yocto
#endif
