#pragma once

#include <yocto/yocto_bvh.h>

#include "boolsurf_utils.h"

namespace yocto {

const static int null_label = -999;

struct scope_timer {
  string  message    = "";
  int64_t start_time = -1;
  bool    print      = true;
  scope_timer(const string& msg);
  ~scope_timer();  // print time when scope ends
};
#define _PROFILE_SCOPE(name) ;
// #define _PROFILE_SCOPE(name) auto _profile = scope_timer(string(name));
#define _PROFILE() _PROFILE_SCOPE(__FUNCTION__)

struct bool_borders {
  vector<bool> tags = {};
};

struct facet {
  std::array<vec2f, 3> corners = {};
  int                  id      = -1;
};

struct mesh_segment {
  vec2f start = {};
  vec2f end   = {};
  int   face  = -1;
};

struct shade_instance;

struct mesh_polygon {
  vector<int>                  points = {};
  vector<vector<mesh_segment>> edges  = {};
  int                          length = 0;

  bool is_contained_in_single_face = false;
  bool is_closed                   = true;
};

struct bool_homotopy_basis {
  int                 root    = -1;
  vector<vector<int>> basis   = {};
  vector<float>       lengths = {};

  vector<mesh_polygon> smooth_basis = {};
};

struct bool_mesh : scene_shape {
  vector<vec3i>        adjacencies    = {};
  vector<vector<int>>  triangle_rings = {};
  dual_geodesic_solver dual_solver    = {};
  bool_borders         borders        = {};

  shape_bvh                    bvh                = {};
  bbox3f                       bbox               = {};
  int                          num_triangles      = 0;
  int                          num_positions      = 0;
  hash_map<int, vector<facet>> triangulated_faces = {};
  geodesic_solver              graph              = {};

  // Shape, Polygon -> Vector({Right face, Left face})
  hash_map<vec2i, vector<vec2i>>    polygon_borders        = {};
  bool_homotopy_basis               homotopy_basis         = {};
  hash_map<vec2i, pair<int, float>> homotopy_basis_borders = {};
  vector<int>                       face_tags              = {};
};

struct shape {
  vector<mesh_polygon> polygons = {};

  vector<int> generators = {};
  bool        is_root    = true;

  vec3f         color = {0, 0, 0};
  hash_set<int> cells = {};

  vector<vector<int>> border_points = {};
  shade_instance*     borders_shape = nullptr;
};

// Informazioni per la triangolazione di una faccia della mesh
// Contiene: UV coords dei nodi locali di un triangolo.
// Indici globali della mesh corrispondenti ai nodi locali
// Edges con indici locali per vincolare la triangolazione
// Mappa che va da lato del triangolo k = 1, 2, 3 e a lista di nodi e lerp
// corrispondenti su quel lato (serve per creare ulteriori vincoli)
struct triangulation_info {
  int face = -1;

  vector<vec2f>                      nodes   = {};
  vector<int>                        indices = {};
  vector<vec2i>                      edges   = {};
  array<vector<pair<int, float>>, 3> edgemap = {};
};

struct mesh_cell {
  vector<int>     faces     = {};
  hash_set<vec2i> adjacency = {};  // {cell_id, crossed_polygon_id}
};

// struct mesh_shape {
//   int         shape      = 0;
//   vector<int> generators = {-1, -1};
//   bool        is_root    = true;

//   vec3f         color = {0, 0, 0};
//   hash_set<int> cells = {};

//   vector<vector<int>> border_points = {};
//   shade_instance*     borders_shape = nullptr;
// };
struct hashgrid_polyline {
  int shape_id   = -1;
  int polygon_id = -1;

  vector<vec2f> points   = {};
  vector<int>   vertices = {};

  bool is_closed = false;
};

using mesh_hashgrid = hash_map<int, vector<hashgrid_polyline>>;

struct bool_state {
  vector<mesh_polygon> polygons    = {{}};
  vector<shape>        bool_shapes = {{}};
  vector<mesh_point>   points      = {};

  int                  num_original_points = 0;
  hash_map<int, int>   control_points      = {};
  hash_map<int, vec2i> isecs_generators    = {};

  vector<mesh_cell>   cells          = {};
  vector<vector<int>> labels         = {};
  hash_set<int>       invalid_shapes = {};
  mesh_hashgrid       hashgrid       = {};

  // vector<int>           ambient_cells = {};
  // vector<vector<vec2i>> cycles        = {};

  // vector<mesh_shape> shapes         = {};
  vector<int> shapes_sorting = {};
  bool        failed         = false;
  bool        correct_input  = false;
};

struct bool_operation {
  enum struct Type {
    op_union,
    op_difference,
    op_intersection,
    op_symmetrical_difference
  };
  int  shape_a = -1;
  int  shape_b = -1;
  Type type    = Type::op_union;

  inline static const auto type_names = vector<string>{"op_union",
      "op_difference", "op_intersection", "op_symmetrical_difference"};
};

void init_mesh(bool_mesh& mesh);
void reset_mesh(bool_mesh& mesh);

void update_polygon(bool_state& state, const bool_mesh& mesh, int polygon_id);
/* bool_homotopy_basis  compute_homotopy_basis(bool_mesh& mesh, int root);
vector<mesh_polygon> smooth_homotopy_basis(
    const bool_homotopy_basis& homotopy_basis, const bool_mesh& mesh,
    bool smooth_generators);
void        compute_homotopy_basis_borders(bool_mesh& mesh);
vector<int> sort_homotopy_basis_around_vertex(
    const bool_mesh& mesh, const bool_homotopy_basis& basis); */
vector<int>              compute_polygonal_schema(const vector<int>& basis);
vector<pair<int, float>> compute_polygon_basis_intersections(
    const mesh_polygon& polygon, bool_mesh& mesh);
vector<int> compute_polygon_word(
    const vector<pair<int, float>>& isecs, const vector<int>& polygonal_schema);
mesh_polygon vectorize_generator_loop(
    bool_state& state, const mesh_polygon& generator_loop, int orientation);
vector<int> compute_strip_from_basis(const vector<int>& base,
    const vector<vector<int>>& triangle_rings, const vector<vec3i>& triangles,
    int root);
/* std::tuple<vector<int>, mesh_point, mesh_point> cleaned_strip(
    const vector<vec3i>& triangles, const vector<vec3f>& positions,
    const vector<vec3i>& adjacencies, const vector<int>& strip,
    const mesh_point& start, const mesh_point& end); */
void remove_loops_from_strip(vector<int>& strip);

void              slice_mesh(bool_mesh& mesh, bool_state& state);
vector<mesh_cell> make_cell_graph(bool_mesh& mesh);
void              compute_cell_labels(bool_state& state, bool non_zero);

bool compute_cells(bool_mesh& mesh, bool_state& state, bool non_zero);
vector<mesh_cell> make_mesh_cells(vector<int>& cell_tags,
    const vector<vec3i>& adjacencies, const vector<bool>& border_tags);

void       compute_shapes(bool_state& state);
void       compute_shape_borders(const bool_mesh& mesh, bool_state& state);
bool_state compute_border_polygons(const bool_state& state);
void       compute_polygon_border_tags(bool_mesh& mesh,
          const vector<vec2i>& polygon_borders, vector<bool>& border_tags);
bool check_polygon_validity(bool_mesh& mesh, int shape_id, int polygon_id);
vector<mesh_point> compute_parallel_loop(
    bool_mesh& mesh, const mesh_polygon& polygon);
void compute_bool_operation(bool_state& state, const bool_operation& op);
void compute_bool_operations(
    bool_state& state, const vector<bool_operation>& ops);

void compute_symmetrical_difference(
    bool_state& state, const vector<int>& shapes);

vector<mesh_segment> mesh_segments(const vector<vec3i>& triangles,
    const vector<int>& strip, const vector<float>& lerps,
    const mesh_point& start, const mesh_point& end);

/* geodesic_path compute_geodesic_path(
    const bool_mesh& mesh, const mesh_point& start, const mesh_point& end); */

mesh_point eval_geodesic_path(
    const bool_mesh& mesh, const geodesic_path& path, float t);

/* void recompute_polygon_segments(const bool_mesh& mesh, const bool_state& state,
    mesh_polygon& polygon, int index = 0); */

/* inline geodesic_path straightest_path(const bool_mesh& mesh,
    const mesh_point& start, const vec2f& direction, float length) {
  return straightest_path(mesh.triangles, mesh.positions, mesh.adjacencies,
      start, direction, length);
}

inline geodesic_path straightest_path(
    const bool_mesh& mesh, const mesh_point& start, const vec2f& coord) {
  auto len = length(coord);
  return straightest_path(mesh.triangles, mesh.positions, mesh.adjacencies,
      start, coord / len, len);
} */

inline vec3f eval_position(const bool_mesh& mesh, const mesh_point& point) {
  return eval_position(mesh.triangles, mesh.positions, point);
}

inline vec3f eval_normal(const bool_mesh& mesh, const mesh_point& point) {
  return eval_normal(mesh.triangles, mesh.normals, point);
}

inline vec3f eval_normal(const bool_mesh& mesh, int face) {
  auto [x, y, z] = mesh.triangles[face];
  return triangle_normal(
      mesh.positions[x], mesh.positions[y], mesh.positions[z]);
}

/* mesh_point intersect_mesh(const bool_mesh& mesh, const shape_bvh& bvh,
    const scene_camera& camera, const vec2f& uv);

inline mesh_point intersect_mesh(
    const bool_mesh& mesh, const scene_camera& camera, const vec2f& uv) {
  return intersect_mesh(mesh, mesh.bvh, camera, uv);
} */

vec3f get_cell_color(const bool_state& state, int cell_id, bool color_shapes);

/*
 *
 *
 *
 *
 *
 *
 *
 *
 *     DEBUGGING STUFF
 *
 */

template <typename F>
static vector<int> flood_fill(const bool_mesh& mesh, const vector<int>& start,
    const int polygon, F&& check) {
  auto visited = vector<bool>(mesh.adjacencies.size(), false);

  auto result = vector<int>();
  auto stack  = start;

  while (!stack.empty()) {
    auto face = stack.back();
    stack.pop_back();

    if (visited[face]) continue;
    visited[face] = true;

    result.push_back(face);

    for (auto neighbor : mesh.adjacencies[face]) {
      if (neighbor < 0 || visited[neighbor])
        continue;
      else if (check(face, -polygon) && check(neighbor, -polygon))
        // Check if "face" is not inner and "neighbor" is outer
        stack.push_back(neighbor);
      else if (check(neighbor, polygon))
        stack.push_back(neighbor);
    }
  }

  return result;
}

template <typename F>
static vector<int> flood_fill(
    const bool_mesh& mesh, const vector<int>& start, F&& check) {
  auto visited = vector<bool>(mesh.adjacencies.size(), false);

  auto result = vector<int>();
  auto stack  = start;

  while (!stack.empty()) {
    auto face = stack.back();
    stack.pop_back();

    if (visited[face]) continue;
    visited[face] = true;

    result.push_back(face);

    for (auto neighbor : mesh.adjacencies[face]) {
      if (neighbor < 0 || visited[neighbor]) continue;
      if (check(face, neighbor)) stack.push_back(neighbor);
    }
  }

  return result;
}

template <typename F>
static void flood_fill_debug(
    const bool_mesh& mesh, const vector<int>& start, F&& check) {
  int face = -1;
  if (debug_stack().empty()) {
    debug_restart() = true;
    return;
  }
  while (!debug_stack().empty()) {
    auto f = debug_stack().back();
    debug_stack().pop_back();
    if (debug_visited()[f]) continue;
    face = f;
    break;
  }
  if (face == -1) return;

  debug_visited()[face] = true;

  debug_result().push_back(face);

  // auto tag = mesh.borders.tags[face];
  // auto adj = mesh.adjacencies[face];
  //  printf("\nfrom %d: tag(%d %d %d) adj(%d %d %d)\n", face, tag[0], tag[1],
  //      tag[2], adj[0], adj[1], adj[2]);

  // for (auto neighbor : mesh.adjacencies[face]) {
  for (int k = 0; k < 3; k++) {
    auto neighbor = mesh.adjacencies[face][k];
    if (neighbor < 0 || debug_visited()[neighbor]) continue;
    if (check(face, k)) {
      debug_stack().push_back(neighbor);
    }
    // auto tag = mesh.borders.tags[neighbor];
    // auto adj = mesh.adjacencies[neighbor];
    //      printf("ok   %d: tag(%d %d %d) adj(%d %d %d)\n", neighbor, tag[0],
    //      tag[1],
    //          tag[2], adj[0], adj[1], adj[2]);
    //    printf("no   %d: tag(%d %d %d) adj(%d %d %d)\n", neighbor, tag[0],
    //    tag[1],
    //        tag[2], adj[0], adj[1], adj[2]);
  }
}

bool path_intersects_segments(const vector<mesh_segment>& path,
    const vector<vector<mesh_segment>>& segment_vector);
void update_segment_vector(const vector<mesh_segment>& path, vector<vector<mesh_segment>>& segment_vector);
vector<mesh_segment> geodesic_path_to_segments(const vector<mesh_point>& path, const dual_geodesic_solver& graph,
    const vector<vec3i>& triangles, const vector<vec3f>& positions, const vector<vec3i>& adjacencies);

}  // namespace yocto