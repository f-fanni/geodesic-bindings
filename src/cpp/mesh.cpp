#include "boolsurf/boolsurf.h"
#include "geometrycentral/numerical/linear_algebra_utilities.h"
#include "geometrycentral/surface/direction_fields.h"
#include "geometrycentral/surface/edge_length_geometry.h"
#include "geometrycentral/surface/flip_geodesics.h"
#include "geometrycentral/surface/heat_method_distance.h"
#include "geometrycentral/surface/manifold_surface_mesh.h"
#include "geometrycentral/surface/mesh_graph_algorithms.h"
#include "geometrycentral/surface/simple_polygon_mesh.h"
#include "geometrycentral/surface/surface_mesh.h"
#include "geometrycentral/surface/surface_mesh_factories.h"
#include "geometrycentral/surface/vector_heat_method.h"
#include "geometrycentral/surface/vertex_position_geometry.h"
#include "geometrycentral/utilities/eigen_interop_helpers.h"
#include "yocto/yocto_shape.h"
#include "yocto_mesh/yocto_mesh.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Eigen/Dense"

namespace py = pybind11;

using namespace geometrycentral;
using namespace geometrycentral::surface;


// For overloaded functions, with C++11 compiler only
template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;


// A wrapper class for the heat method solver, which exposes Eigen in/out
class HeatMethodDistanceEigen {

public:
  HeatMethodDistanceEigen(DenseMatrix<double> verts, DenseMatrix<int64_t> faces, double tCoef = 1.0,
                          bool useRobustLaplacian = true) {

    // Construct the internal mesh and geometry
    mesh.reset(new SurfaceMesh(faces));
    geom.reset(new VertexPositionGeometry(*mesh));
    for (size_t i = 0; i < mesh->nVertices(); i++) {
      for (size_t j = 0; j < 3; j++) {
        geom->inputVertexPositions[i][j] = verts(i, j);
      }
    }

    // Build the solver
    solver.reset(new HeatMethodDistanceSolver(*geom, tCoef, useRobustLaplacian));
  }

  // Solve for distance from a single vertex
  Vector<double> compute_distance(int64_t sourceVert) {
    VertexData<double> dist = solver->computeDistance(mesh->vertex(sourceVert));
    return dist.toVector();
  }

  // Solve for distance from a collection of vertices
  Vector<double> compute_distance_multisource(Vector<int64_t> sourceVerts) {
    std::vector<Vertex> sources;
    for (size_t i = 0; i < sourceVerts.rows(); i++) {
      sources.push_back(mesh->vertex(sourceVerts(i)));
    }
    VertexData<double> dist = solver->computeDistance(sources);
    return dist.toVector();
  }

  // Solve for distance from a collection of surface points
  Vector<double> compute_distance_multisource_meshpoint(DenseMatrix<double> barycentric_coords,
                                                        Vector<int64_t> face_ids) {
    std::vector<SurfacePoint> sources;
    // Convert the input to the expected types
    for (size_t i = 0; i < barycentric_coords.rows(); i++) {
      sources.push_back(SurfacePoint(mesh->face(face_ids[i]),
                                     {barycentric_coords(i, 0), barycentric_coords(i, 1), barycentric_coords(i, 2)}));
    }
    VertexData<double> dist = solver->computeDistance(sources);
    return dist.toVector();
  }

private:
  std::unique_ptr<SurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::unique_ptr<HeatMethodDistanceSolver> solver;
};


// A wrapper class for the vector heat method solver, which exposes Eigen in/out
class VectorHeatMethodEigen {

  // TODO use intrinsic triangulations here

public:
  VectorHeatMethodEigen(DenseMatrix<double> verts, DenseMatrix<int64_t> faces, double tCoef = 1.0) {

    // Construct the internal mesh and geometry
    mesh.reset(new ManifoldSurfaceMesh(faces));
    geom.reset(new VertexPositionGeometry(*mesh));
    for (size_t i = 0; i < mesh->nVertices(); i++) {
      for (size_t j = 0; j < 3; j++) {
        geom->inputVertexPositions[i][j] = verts(i, j);
      }
    }

    // Build the solver
    solver.reset(new VectorHeatMethodSolver(*geom, tCoef));
  }

  // Extend scalars from a collection of vertices
  Vector<double> extend_scalar(Vector<int64_t> sourceVerts, Vector<double> values) {
    std::vector<std::tuple<Vertex, double>> sources;
    for (size_t i = 0; i < sourceVerts.rows(); i++) {
      sources.emplace_back(mesh->vertex(sourceVerts(i)), values(i));
    }
    VertexData<double> ext = solver->extendScalar(sources);
    return ext.toVector();
  }


  // Returns an extrinsic representation of the tangent frame being used internally, as X/Y/N vectors.
  std::tuple<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>> get_tangent_frames() {

    // Just in case we don't already have it
    geom->requireVertexNormals();
    geom->requireVertexTangentBasis();

    // unpack
    VertexData<Vector3> basisX(*mesh);
    VertexData<Vector3> basisY(*mesh);
    for (Vertex v : mesh->vertices()) {
      basisX[v] = geom->vertexTangentBasis[v][0];
      basisY[v] = geom->vertexTangentBasis[v][1];
    }

    return std::tuple<DenseMatrix<double>, DenseMatrix<double>, DenseMatrix<double>>(
        EigenMap<double, 3>(basisX), EigenMap<double, 3>(basisY), EigenMap<double, 3>(geom->vertexNormals));
  }

  SparseMatrix<std::complex<double>> get_connection_laplacian() {
    geom->requireVertexConnectionLaplacian();
    SparseMatrix<std::complex<double>> Lconn = geom->vertexConnectionLaplacian;
    geom->unrequireVertexConnectionLaplacian();
    return Lconn;
  }

  // TODO think about how to pass tangent frames around
  DenseMatrix<double> transport_tangent_vectors(Vector<int64_t> sourceVerts, DenseMatrix<double> values) {

    // Pack it as a Vector2
    std::vector<std::tuple<Vertex, Vector2>> sources;
    for (size_t i = 0; i < sourceVerts.rows(); i++) {
      sources.emplace_back(mesh->vertex(sourceVerts(i)), Vector2{values(i, 0), values(i, 1)});
    }
    VertexData<Vector2> ext = solver->transportTangentVectors(sources);

    return EigenMap<double, 2>(ext);
  }

  DenseMatrix<double> transport_tangent_vector(int64_t sourceVert, DenseMatrix<double> values) {

    // Pack it as a Vector2
    std::vector<std::tuple<Vertex, Vector2>> sources;
    sources.emplace_back(mesh->vertex(sourceVert), Vector2{values(0), values(1)});
    VertexData<Vector2> ext = solver->transportTangentVectors(sources);

    return EigenMap<double, 2>(ext);
  }


  DenseMatrix<double> compute_log_map(int64_t sourceVert) {
    return EigenMap<double, 2>(solver->computeLogMap(mesh->vertex(sourceVert)));
  }

private:
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::unique_ptr<VectorHeatMethodSolver> solver;
};

// A wrapper class for flip-based geodesics
class EdgeFlipGeodesicsManager {

public:
  EdgeFlipGeodesicsManager(DenseMatrix<double> verts, DenseMatrix<int64_t> faces) {

    // Construct the internal mesh and geometry
    mesh.reset(new ManifoldSurfaceMesh(faces));
    geom.reset(new VertexPositionGeometry(*mesh));
    for (size_t i = 0; i < mesh->nVertices(); i++) {
      for (size_t j = 0; j < 3; j++) {
        geom->inputVertexPositions[i][j] = verts(i, j);
      }
    }

    // Build the solver
    flipNetwork.reset(new FlipEdgeNetwork(*mesh, *geom, {}));
    flipNetwork->posGeom = geom.get();
    flipNetwork->supportRewinding = true;
  }

  // Generate a point-to-point geodesic by straightening a Dijkstra path
  DenseMatrix<double> find_geodesic_path(int64_t startVert, int64_t endVert) {

    // Get an initial dijkstra path
    std::vector<Halfedge> dijkstraPath = shortestEdgePath(*geom, mesh->vertex(startVert), mesh->vertex(endVert));

    if (startVert == endVert) {
      throw std::runtime_error("start and end vert are same");
    }
    if (dijkstraPath.empty()) {
      throw std::runtime_error("vertices lie on disconnected components of the surface");
    }

    // Reinitialize the ede network to contain this path
    flipNetwork->reinitializePath({dijkstraPath});

    // Straighten the path to geodesic
    flipNetwork->iterativeShorten();

    // Extract the path and store it in the vector
    std::vector<Vector3> path3D = flipNetwork->getPathPolyline3D().front();
    DenseMatrix<double> out(path3D.size(), 3);
    for (size_t i = 0; i < path3D.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        out(i, j) = path3D[i][j];
      }
    }

    // Be kind, rewind
    flipNetwork->rewind();

    return out;
  }

  // Generate a point-to-point geodesic by straightening a poly-geodesic path
  DenseMatrix<double> find_geodesic_path_poly(std::vector<int64_t> verts) {

    // Convert to a list of vertices
    std::vector<Halfedge> halfedges;

    for (size_t i = 0; i + 1 < verts.size(); i++) {
      Vertex vA = mesh->vertex(verts[i]);
      Vertex vB = mesh->vertex(verts[i + 1]);
      std::vector<Halfedge> dijkstraPath = shortestEdgePath(*geom, vA, vB);

      // validate
      if (vA == vB) {
        throw std::runtime_error("consecutive vertices are same");
      }
      if (dijkstraPath.empty()) {
        throw std::runtime_error("vertices lie on disconnected components of the surface");
      }

      halfedges.insert(halfedges.end(), dijkstraPath.begin(), dijkstraPath.end());
    }

    // Reinitialize the ede network to contain this path
    flipNetwork->reinitializePath({halfedges});

    // Straighten the path to geodesic
    flipNetwork->iterativeShorten();

    // Extract the path and store it in the vector
    std::vector<Vector3> path3D = flipNetwork->getPathPolyline3D().front();
    DenseMatrix<double> out(path3D.size(), 3);
    for (size_t i = 0; i < path3D.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        out(i, j) = path3D[i][j];
      }
    }

    // Be kind, rewind
    flipNetwork->rewind();

    return out;
  }


  // Generate a point-to-point geodesic loop by straightening a poly-geodesic path
  DenseMatrix<double> find_geodesic_loop(std::vector<int64_t> verts) {

    // Convert to a list of vertices
    std::vector<Halfedge> halfedges;

    for (size_t i = 0; i < verts.size(); i++) {
      Vertex vA = mesh->vertex(verts[i]);
      Vertex vB = mesh->vertex(verts[(i + 1) % verts.size()]);
      std::vector<Halfedge> dijkstraPath = shortestEdgePath(*geom, vA, vB);

      // validate
      if (vA == vB) {
        throw std::runtime_error("consecutive vertices are same");
      }
      if (dijkstraPath.empty()) {
        throw std::runtime_error("vertices lie on disconnected components of the surface");
      }

      halfedges.insert(halfedges.end(), dijkstraPath.begin(), dijkstraPath.end());
    }

    // Reinitialize the ede network to contain this path
    flipNetwork->reinitializePath({halfedges});

    // Straighten the path to geodesic
    flipNetwork->iterativeShorten();

    // Extract the path and store it in the vector
    std::vector<Vector3> path3D = flipNetwork->getPathPolyline3D().front();
    DenseMatrix<double> out(path3D.size(), 3);
    for (size_t i = 0; i < path3D.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        out(i, j) = path3D[i][j];
      }
    }

    // Be kind, rewind
    flipNetwork->rewind();

    return out;
  }

  // Generate a bezier by straightening a poly-geodesic path
  DenseMatrix<double> compute_bezier_curve(std::vector<int64_t> verts, int64_t nRounds) {

    // Convert to a list of vertices
    std::vector<Vertex> vertices(verts.size());

    for (size_t i = 0; i < verts.size(); i++) {
      vertices[i] = mesh->vertex(verts[i]);
    }

    // Construct a suitable network for bezier
    auto bezierNetwork = FlipEdgeNetwork::constructFromPiecewiseDijkstraPath(*mesh, *geom, vertices, false, true);
    bezierNetwork->posGeom = geom.get();

    // Create the bezier
    bezierNetwork->bezierSubdivide(nRounds);

    // Extract the path and store it in the vector
    std::vector<Vector3> path3D = bezierNetwork->getPathPolyline3D().front();
    DenseMatrix<double> out(path3D.size(), 3);
    for (size_t i = 0; i < path3D.size(); i++) {
      for (size_t j = 0; j < 3; j++) {
        out(i, j) = path3D[i][j];
      }
    }

    // Be kind, rewind
    // bezierNetwork->rewind();
    // No rewind necessary, bezierNetwork will be destroyed anyway

    return out;
  }

  // Generate a bezier by straightening a poly-geodesic path, returned as a geodesic path
  std::pair<DenseMatrix<double>, Vector<int>> compute_bezier_curve_geopath(std::vector<int64_t> verts,
                                                                           int64_t nRounds) {

    // Convert to a list of vertices
    std::vector<Vertex> vertices(verts.size());

    for (size_t i = 0; i < verts.size(); i++) {
      vertices[i] = mesh->vertex(verts[i]);
    }

    // Construct a suitable network for bezier
    auto bezierNetwork = FlipEdgeNetwork::constructFromPiecewiseDijkstraPath(*mesh, *geom, vertices, false, true);
    bezierNetwork->posGeom = geom.get();

    // Create the bezier
    bezierNetwork->bezierSubdivide(nRounds);

    // Extract the path and store it in the vector
    auto path = bezierNetwork->getPathPolyline().front();
    DenseMatrix<double> barycentric_coords(path.size(), 3);
    Vector<int> face_indices(path.size());
    for (size_t i = 0; i < path.size(); i++) {
      auto face_point = path[i].inSomeFace();
      face_indices[i] = face_point.face.getIndex();
      for (size_t j = 0; j < 3; j++) {
        barycentric_coords(i, j) = face_point.faceCoords[j];
      }
    }

    // Be kind, rewind
    // bezierNetwork->rewind();
    // No rewind necessary, bezierNetwork will be destroyed anyway

    return {barycentric_coords, face_indices};
  }

private:
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  std::unique_ptr<FlipEdgeNetwork> flipNetwork;
};


// A wrapper class for tracing straight geodesics
// The method itself is not stateful, but this avoids passing vertices and faces every time
class TraceGeodesicsMethod {

public:
  TraceGeodesicsMethod(DenseMatrix<double> verts, DenseMatrix<int64_t> faces) {

    // Construct the internal mesh and geometry
    mesh.reset(new ManifoldSurfaceMesh(faces));
    geom.reset(new VertexPositionGeometry(*mesh));
    for (size_t i = 0; i < mesh->nVertices(); i++) {
      for (size_t j = 0; j < 3; j++) {
        geom->inputVertexPositions[i][j] = verts(i, j);
      }
    }
  }

  // Trace a geodesic path with a given direction, returns the final point, the face it lies in and the ending direction
  std::tuple<Eigen::Vector3d, int64_t, Eigen::Vector2d> trace_geodesic_path(int64_t startVertex,
                                                                            DenseMatrix<double> traceVec) {

    // Convert the input to the expected types
    Vector2 direction({traceVec(0), traceVec(1)});
    SurfacePoint startPoint(mesh->vertex(startVertex));

    // Actual path tracing
    auto res = traceGeodesic(*geom, startPoint, direction);

    // Extract the data to return
    Vector3 endP = res.endPoint.interpolate(geom->vertexPositions);
    int64_t faceInd = res.endPoint.inSomeFace().face.getIndex();
    Vector2 endDir = res.endingDir;
    return std::tuple<Eigen::Vector3d, int64_t, Eigen::Vector2d>(Eigen::Vector3d({endP.x, endP.y, endP.z}), faceInd,
                                                                 Eigen::Vector2d({endDir.x, endDir.y}));
  }

  // Trace a geodesic path with a given direction, returns the final point as meshpoint, the face it lies in and the
  // ending direction and whether we hit a boundary
  std::tuple<Eigen::Vector3d, int64_t, Eigen::Vector2d, bool>
  trace_geodesic_path_meshpoint(DenseMatrix<double> barycentric_coords, int64_t face_id, DenseMatrix<double> traceVec) {

    // Convert the input to the expected types
    Vector2 direction({traceVec(0), traceVec(1)});
    SurfacePoint startPoint(mesh->face(face_id),
                            Vector3({barycentric_coords(0), barycentric_coords(1), barycentric_coords(2)}));

    // Actual path tracing
    auto res = traceGeodesic(*geom, startPoint, direction);

    // Extract the data to return
    SurfacePoint res_point = res.endPoint.inSomeFace();
    Vector3 endP = res_point.faceCoords;
    int64_t faceInd = res_point.face.getIndex();
    Vector2 endDir = res.endingDir;
    return std::tuple<Eigen::Vector3d, int64_t, Eigen::Vector2d, bool>(
        Eigen::Vector3d({endP.x, endP.y, endP.z}), faceInd, Eigen::Vector2d({endDir.x, endDir.y}), res.hitBoundary);
  }

private:
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
};

DenseMatrix<double> compute_direction_field(DenseMatrix<double> verts, DenseMatrix<int64_t> faces, int n_sym) {
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  DenseMatrix<double> res(verts.rows(), 2 * n_sym);
  // Construct the internal mesh and geometry
  mesh.reset(new ManifoldSurfaceMesh(faces));
  geom.reset(new VertexPositionGeometry(*mesh));
  for (size_t i = 0; i < mesh->nVertices(); i++) {
    for (size_t j = 0; j < 3; j++) {
      geom->inputVertexPositions[i][j] = verts(i, j);
    }
  }
  // Compute the field
  VertexData<Vector2> directions = computeSmoothestVertexDirectionField(*geom, n_sym);
  // Generate the explicit vectors in the tangent plane
  for (int i = 0; i < mesh->nVertices(); i++) {
    const auto& v = mesh->vertex(i);
    Vector2 representative = directions[v];
    Vector2 crossDir = representative.pow(1. / n_sym); // take the n'th root

    // loop over the four directions
    for (int rot = 0; rot < n_sym; rot++) {
      // crossDir is one of the four cross directions, as a tangent vector
      crossDir = crossDir.rotate((2 * M_PI / n_sym) * rot);
      res(i, 2 * rot) = crossDir.x;
      res(i, (2 * rot) + 1) = crossDir.y;
    }
  }
  return res;
}

DenseMatrix<double> compute_face_direction_field(DenseMatrix<double> verts, DenseMatrix<int64_t> faces, int n_sym) {
  std::unique_ptr<ManifoldSurfaceMesh> mesh;
  std::unique_ptr<VertexPositionGeometry> geom;
  DenseMatrix<double> res(faces.rows(), 2 * n_sym);
  // Construct the internal mesh and geometry
  mesh.reset(new ManifoldSurfaceMesh(faces));
  geom.reset(new VertexPositionGeometry(*mesh));
  for (size_t i = 0; i < mesh->nVertices(); i++) {
    for (size_t j = 0; j < 3; j++) {
      geom->inputVertexPositions[i][j] = verts(i, j);
    }
  }
  // Compute the field
  FaceData<Vector2> directions = computeSmoothestFaceDirectionField(*geom, n_sym);
  // Generate the explicit vectors in the tangent plane
  for (int i = 0; i < mesh->nFaces(); i++) {
    const auto& f = mesh->face(i);
    Vector2 representative = directions[f];
    Vector2 crossDir = representative.pow(1. / n_sym); // take the n'th root

    // loop over the four directions
    for (int rot = 0; rot < n_sym; rot++) {
      // crossDir is one of the four cross directions, as a tangent vector
      crossDir = crossDir.rotate((2 * M_PI / n_sym) * rot);
      res(i, 2 * rot) = crossDir.x;
      res(i, (2 * rot) + 1) = crossDir.y;
    }
  }
  return res;
}

// A wrapper class for the yocto mesh methods
class YoctoMeshManager {

public:
  YoctoMeshManager(DenseMatrix<double> verts, DenseMatrix<int64_t> faces) {
    positions.resize(verts.rows());
    for (int i = 0; i < positions.size(); i++) {
      positions[i] = {static_cast<float>(verts(i, 0)), static_cast<float>(verts(i, 1)),
                      static_cast<float>(verts(i, 2))};
    }
    triangles.resize(faces.rows());
    for (int i = 0; i < triangles.size(); i++) {
      triangles[i] = {static_cast<int>(faces(i, 0)), static_cast<int>(faces(i, 1)), static_cast<int>(faces(i, 2))};
    }
    adjacencies = yocto::face_adjacencies(triangles);
    geo_solver = yocto::make_geodesic_solver(triangles, adjacencies, positions);
    dual_geo_solver = yocto::make_dual_geodesic_solver(triangles, positions, adjacencies);
    // vertices2faces = yocto::vertex_to_faces_adjacencies(triangles, adjacencies);
    vertices2mesh_point.resize(triangles.size());
    for (int i = 0; i < triangles.size(); i++) {
      vertices2mesh_point[triangles[i][0]] = {i, {0, 0}};
      vertices2mesh_point[triangles[i][1]] = {i, {1, 0}};
      vertices2mesh_point[triangles[i][2]] = {i, {0, 1}};
    }
    // vector of segments
    placed_segments = std::vector<std::vector<yocto::mesh_segment>>(triangles.size());
  }

  std::pair<DenseMatrix<double>, Vector<int>> compute_bezier_curve_vertices(std::vector<int64_t> verts,
                                                                            int64_t subdivisions = 4) {
    DenseMatrix<double> res_coords;
    Vector<int> res_faces;
    std::vector<yocto::mesh_point> bezier_path;

    std::vector<yocto::mesh_point> controls(verts.size());
    for (int i = 0; i < verts.size(); i++) {
      controls[i] = vertices2mesh_point[verts[i]];
    }
    bezier_path =
        yocto::compute_bezier_uniform(dual_geo_solver, triangles, positions, adjacencies, controls, subdivisions);

    res_coords.resize(bezier_path.size(), 3);
    res_faces.resize(bezier_path.size());
    for (int i = 0; i < bezier_path.size(); i++) {
      auto&& mp = bezier_path[i];
      res_faces(i) = mp.face;
      res_coords(i, 0) = (1 - (mp.uv[0] + mp.uv[1]));
      res_coords(i, 1) = mp.uv[0];
      res_coords(i, 2) = mp.uv[1];
    }
    return {res_coords, res_faces};
  }

  std::pair<DenseMatrix<double>, Vector<int>> compute_bezier_curve_meshpoints(DenseMatrix<double> barycentric_coords,
                                                                              std::vector<int64_t> face_ids,
                                                                              int64_t subdivisions = 4) {
    DenseMatrix<double> res_coords;
    Vector<int> res_faces;
    std::vector<yocto::mesh_point> bezier_path;
    std::vector<yocto::mesh_point> control_points;

    std::vector<yocto::mesh_point> controls(face_ids.size());
    for (int i = 0; i < face_ids.size(); i++) {
      controls[i] = {static_cast<int>(face_ids[i]),
                     {static_cast<float>(barycentric_coords(i, 1)), static_cast<float>(barycentric_coords(i, 2))}};
    }
    control_points =
        yocto::compute_bezier_uniform(dual_geo_solver, triangles, positions, adjacencies, controls, subdivisions);
    for (int i = 0; i < control_points.size() - 1; i++) {
      auto path =
          yocto::convert_mesh_path(triangles, adjacencies,
                                   yocto::compute_shortest_path(dual_geo_solver, triangles, positions, adjacencies,
                                                                control_points[i], control_points[i + 1]));
      bezier_path.insert(bezier_path.end(), path.begin(), path.end());
    }
    res_coords.resize(bezier_path.size(), 3);
    res_faces.resize(bezier_path.size());
    for (int i = 0; i < bezier_path.size(); i++) {
      auto&& mp = bezier_path[i];
      res_faces(i) = mp.face;
      res_coords(i, 0) = (1 - (mp.uv[0] + mp.uv[1]));
      res_coords(i, 1) = mp.uv[0];
      res_coords(i, 2) = mp.uv[1];
    }
    return {res_coords, res_faces};
  }

  // Solve for distance from a collection of vertices
  std::vector<double> compute_distance(std::vector<int64_t> sourceVerts) {
    std::vector<int> aux(sourceVerts.begin(), sourceVerts.end());
    std::vector<float> res = compute_geodesic_distances(geo_solver, aux);

    return std::vector<double>(res.begin(), res.end());
  }

  // Solve for distance from a collection of meshpoints
  // TODO this is probably not working as inteded
  Vector<double> compute_distance_meshpoints(DenseMatrix<double> barycentric_coords, Vector<int64_t> face_ids,
                                             float max_distance = yocto::flt_max) {
    auto distances = std::vector<float>(geo_solver.graph.size(), yocto::flt_max);
    std::vector<int> sources;
    sources.reserve(2 * face_ids.size()); // just a rough estimate
    for (int i = 0; i < face_ids.size(); i++) {
      int f_i = face_ids[i];
      auto point = yocto::interpolate_triangle(
          positions[triangles[f_i][0]], positions[triangles[f_i][1]], positions[triangles[f_i][2]],
          {static_cast<float>(barycentric_coords(1)), static_cast<float>(barycentric_coords(2))});
      for (auto j : {0, 1, 2}) {
        distances[triangles[f_i][j]] = yocto::distance(positions[triangles[f_i][j]], point);
        sources.push_back(static_cast<int>(triangles[f_i][j]));
      }
    }
    yocto::update_geodesic_distances(distances, geo_solver, sources, max_distance);
    auto aux = std::vector<double>(distances.begin(), distances.end());
    return Eigen::Map<Vector<double>, Eigen::Unaligned>(aux.data(), aux.size());
  }

  bool path_intersect_others(DenseMatrix<double> barycentric_coords, Vector<int> face_ids) {
    std::vector<yocto::mesh_point> points(face_ids.size());
    for (int i = 0; i < face_ids.size(); i++) {
      points[i] = {static_cast<int>(face_ids[i]),
                   {static_cast<float>(barycentric_coords(i, 1)), static_cast<float>(barycentric_coords(i, 2))}};
    }
    auto segments = yocto::geodesic_path_to_segments(points, dual_geo_solver, triangles, positions, adjacencies);
    return yocto::path_intersects_segments(segments, placed_segments);
  }

  void update_stored_paths(DenseMatrix<double> barycentric_coords, Vector<int> face_ids) {
    std::vector<yocto::mesh_point> points(face_ids.size());
    for (int i = 0; i < face_ids.size(); i++) {
      points[i] = {static_cast<int>(face_ids[i]),
                   {static_cast<float>(barycentric_coords(i, 1)), static_cast<float>(barycentric_coords(i, 2))}};
    }
    auto segments = yocto::geodesic_path_to_segments(points, dual_geo_solver, triangles, positions, adjacencies);
    yocto::update_segment_vector(segments, placed_segments);
    return;
  }

private:
  std::vector<yocto::vec3i> triangles;
  std::vector<yocto::vec3i> adjacencies;
  std::vector<yocto::vec3f> positions;
  std::vector<yocto::mesh_point> vertices2mesh_point;
  yocto::geodesic_solver geo_solver;
  yocto::dual_geodesic_solver dual_geo_solver;
  std::vector<std::vector<yocto::mesh_segment>> placed_segments;
};

// Actual binding code
// clang-format off
void bind_mesh(py::module& m) {

  py::class_<HeatMethodDistanceEigen>(m, "MeshHeatMethodDistance")
        .def(py::init<DenseMatrix<double>, DenseMatrix<int64_t>, double, bool>())
        .def("compute_distance", &HeatMethodDistanceEigen::compute_distance, py::arg("source_vert"))
        .def("compute_distance_multisource", &HeatMethodDistanceEigen::compute_distance_multisource, py::arg("source_verts"))
        .def("compute_distance_multisource_meshpoint", &HeatMethodDistanceEigen::compute_distance_multisource_meshpoint, py::arg("barycentric_coords"), py::arg("face_ids"));
 

  py::class_<VectorHeatMethodEigen>(m, "MeshVectorHeatMethod")
        .def(py::init<DenseMatrix<double>, DenseMatrix<int64_t>, double>())
        .def("extend_scalar", &VectorHeatMethodEigen::extend_scalar, py::arg("source_verts"), py::arg("values"))
        .def("get_tangent_frames", &VectorHeatMethodEigen::get_tangent_frames)
        .def("get_connection_laplacian", &VectorHeatMethodEigen::get_connection_laplacian)
        .def("transport_tangent_vector", &VectorHeatMethodEigen::transport_tangent_vector, py::arg("source_vert"), py::arg("vector"))
        .def("transport_tangent_vectors", &VectorHeatMethodEigen::transport_tangent_vectors, py::arg("source_verts"), py::arg("vectors"))
        .def("compute_log_map", &VectorHeatMethodEigen::compute_log_map, py::arg("source_vert"));


  py::class_<EdgeFlipGeodesicsManager>(m, "EdgeFlipGeodesicsManager")
        .def(py::init<DenseMatrix<double>, DenseMatrix<int64_t>>())
        .def("find_geodesic_path", &EdgeFlipGeodesicsManager::find_geodesic_path, py::arg("source_vert"), py::arg("target_vert"))
        .def("find_geodesic_path_poly", &EdgeFlipGeodesicsManager::find_geodesic_path_poly, py::arg("vert_list"))
        .def("find_geodesic_loop", &EdgeFlipGeodesicsManager::find_geodesic_loop, py::arg("vert_list"))
        .def("compute_bezier_curve", &EdgeFlipGeodesicsManager::compute_bezier_curve, py::arg("vert_list"), py::arg("n_rounds"))
        .def("compute_bezier_curve_geopath", &EdgeFlipGeodesicsManager::compute_bezier_curve_geopath, py::arg("vert_list"), py::arg("n_rounds"));

  py::class_<TraceGeodesicsMethod>(m, "TraceGeodesicsMethod")
        .def(py::init<DenseMatrix<double>, DenseMatrix<int64_t>>())
        .def("trace_geodesic_path", &TraceGeodesicsMethod::trace_geodesic_path, py::arg("start_vertex"), py::arg("trace_vector"))
        .def("trace_geodesic_meshpoint", &TraceGeodesicsMethod::trace_geodesic_path_meshpoint, py::arg("barycentric_coords"), py::arg("face_ids"), py::arg("trace_vector"));

  m.def("compute_direction_field", &compute_direction_field, py::arg("vert_list"), py::arg("face_list"), py::arg("n_symmetries"));
  m.def("compute_face_direction_field", &compute_face_direction_field, py::arg("vert_list"), py::arg("face_list"), py::arg("n_symmetries"));

  py::class_<YoctoMeshManager>(m, "YoctoMeshManager")
        .def(py::init<DenseMatrix<double>, DenseMatrix<int64_t>>())
        .def("compute_bezier_curve_vertices", &YoctoMeshManager::compute_bezier_curve_vertices, py::arg("vert_list"), py::arg("subdivisions"))
        .def("compute_bezier_curve_meshpoints", &YoctoMeshManager::compute_bezier_curve_meshpoints, py::arg("barycentric_coords"), py::arg("face_ids"), py::arg("subdivisions"))
        .def("compute_distance", &YoctoMeshManager::compute_distance, py::arg("sourceVerts"))
        .def("compute_distance_meshpoints", &YoctoMeshManager::compute_distance_meshpoints, py::arg("barycentric_coords"), py::arg("face_ids"), py::arg("max_distance"))
        .def("path_intersect_others", &YoctoMeshManager::path_intersect_others, py::arg("barycentric_coords"), py::arg("face_ids"))
        .def("update_stored_paths", &YoctoMeshManager::update_stored_paths, py::arg("barycentric_coords"), py::arg("face_ids"));

  //m.def("read_mesh", &read_mesh, "Read a mesh from file.", py::arg("filename"));
}
