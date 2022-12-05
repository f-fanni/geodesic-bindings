import numpy as np
import potpourri3d_bindings as pp3db

import scipy
import scipy.sparse

from .core import *

class MeshHeatMethodDistanceSolver():

    def __init__(self, V, F, t_coef=1., use_robust=True):
        validate_mesh(V, F, force_triangular=True, test_indices=True)
        self.bound_solver = pp3db.MeshHeatMethodDistance(V, F, t_coef, use_robust)

    def compute_distance(self, v_ind):
        return self.bound_solver.compute_distance(v_ind)
    
    def compute_distance_multisource(self, v_inds):
        return self.bound_solver.compute_distance_multisource(v_inds)

    def compute_distance_multisource_meshpoint(self, barycentric_coords, face_ids):
        return self.bound_solver.compute_distance_multisource_meshpoint(barycentric_coords, face_ids)

def compute_distance(V, F, v_ind): 
    solver = MeshHeatMethodDistanceSolver(V, F)
    return solver.compute_distance(v_ind)

def compute_distance_multisource(V, F, v_inds):
    solver = MeshHeatMethodDistanceSolver(V, F)
    return solver.compute_distance_multisource(v_inds)

def compute_distance_multisource_meshpoint(V, F, barycentric_coords, face_ids):
    solver = MeshHeatMethodDistanceSolver(V, F)
    return solver.compute_distance_multisource_meshpoint(barycentric_coords, face_ids)


class MeshVectorHeatSolver():

    def __init__(self, V, F, t_coef=1.):
        validate_mesh(V, F, force_triangular=True, test_indices=True)
        self.bound_solver = pp3db.MeshVectorHeatMethod(V, F, t_coef)

    def extend_scalar(self, v_inds, values):
        if len(v_inds) != len(values):
            raise ValueError("source vertex indices and values array should be same shape")
        return self.bound_solver.extend_scalar(v_inds, values)
    
    def get_tangent_frames(self):
        return self.bound_solver.get_tangent_frames()
    
    def get_connection_laplacian(self):
        return self.bound_solver.get_connection_laplacian()
    
    def transport_tangent_vector(self, v_ind, vector):
        if len(vector) != 2:
            raise ValueError("vector should be a 2D tangent vector")
        return self.bound_solver.transport_tangent_vector(v_ind, vector)
    
    def transport_tangent_vectors(self, v_inds, vectors):
        if len(v_inds) != len(vectors):
            raise ValueError("source vertex indices and values array should be same length")
        return self.bound_solver.transport_tangent_vectors(v_inds, vectors)
    
    def compute_log_map(self, v_ind):
        return self.bound_solver.compute_log_map(v_ind)

class EdgeFlipGeodesicSolver():

    def __init__(self, V, F, t_coef=1.):
        validate_mesh(V, F, force_triangular=True)
        self.bound_solver = pp3db.EdgeFlipGeodesicsManager(V, F)

    def find_geodesic_path(self, v_start, v_end):
        return self.bound_solver.find_geodesic_path(v_start, v_end)
    
    def find_geodesic_path_poly(self, v_list):
        return self.bound_solver.find_geodesic_path_poly(v_list)
    
    def find_geodesic_loop(self, v_list):
        return self.bound_solver.find_geodesic_loop(v_list)

    def compute_bezier_curve(self, v_list, n_rounds):
        return self.bound_solver.compute_bezier_curve(v_list, n_rounds)

    def compute_bezier_curve_geopath(self, v_list, n_rounds):
        return self.bound_solver.compute_bezier_curve_geopath(v_list, n_rounds)


class TraceGeodesicsSolver():

    def __init__(self, V, F):
        validate_mesh(V, F, force_triangular=True)
        self.bound_solver = pp3db.TraceGeodesicsMethod(V, F)

    def trace_geodesic(self, v_ind, vector):
        if len(vector) != 2:
            raise ValueError("vector should be a 2D tangent vector")
        return self.bound_solver.trace_geodesic_path(v_ind, vector)

    def trace_geodesic_meshpoint(self, barycentric_coords, face_ind, vector):
        if len(vector) != 2:
            raise ValueError("vector should be a 2D tangent vector")
        if len(barycentric_coords) != 3:
            raise ValueError("barycentric_coords should contain 3 values")

        return self.bound_solver.trace_geodesic_meshpoint(barycentric_coords, face_ind, vector)

class YoctoMeshSolver():

    def __init__(self, V, F):
        self.bound_solver = pp3db.YoctoMeshManager(V, F)

    def compute_bezier_curve_vertices(self, v_list, subdivisions=4):
        return self.bound_solver.compute_bezier_curve_vertices(v_list, subdivisions)

    def compute_bezier_curve_meshpoints(self, barycentric_coordinates, face_ids, subdivisions=4):
        return self.bound_solver.compute_bezier_curve_meshpoints(barycentric_coordinates, face_ids, subdivisions)

    def compute_distance(self, sourceVerts):
        return np.array(self.bound_solver.compute_distance(sourceVerts))

    def compute_distance_meshpoints(self, barycentric_coordinates, face_ids, max_distance):
        return (self.bound_solver.compute_distance_meshpoints(barycentric_coordinates, face_ids, max_distance))

def compute_direction_field(V, F, n_symmetries):
    return pp3db.compute_direction_field(V, F, n_symmetries)

def compute_face_direction_field(V, F, n_symmetries):
    return pp3db.compute_face_direction_field(V, F, n_symmetries)

def cotan_laplacian(V, F, denom_eps=0.):
    validate_mesh(V, F, force_triangular=True)
    nV = V.shape[0]

    mat_i = []
    mat_j = []
    mat_data = []
    for i in range(3):
      
        # Gather indices and compute cotan weight (via dot() / cross() formula)
        inds_i = F[:,i]
        inds_j = F[:,(i+1)%3]
        inds_k = F[:,(i+2)%3]
        vec_ki = V[inds_i,:] - V[inds_k,:]
        vec_kj = V[inds_j,:] - V[inds_k,:]
        dots = np.sum(vec_ki * vec_kj, axis=1)
        cross_mags =  np.linalg.norm(np.cross(vec_ki, vec_kj), axis=1)
        cotans = 0.5 * dots / (cross_mags + denom_eps)

        # Add the four matrix entries from this weight

        mat_i.append(inds_i)
        mat_j.append(inds_i)
        mat_data.append(cotans)
        
        mat_i.append(inds_j)
        mat_j.append(inds_j)
        mat_data.append(cotans)
        
        mat_i.append(inds_i)
        mat_j.append(inds_j)
        mat_data.append(-cotans)
        
        mat_i.append(inds_j)
        mat_j.append(inds_i)
        mat_data.append(-cotans)

    # Concatenate the arrays to single lists
    mat_i = np.concatenate(mat_i)
    mat_j = np.concatenate(mat_j)
    mat_data = np.concatenate(mat_data)

    L_coo = scipy.sparse.coo_matrix((mat_data, (mat_i, mat_j)), shape=(nV, nV))

    return L_coo.tocsr()

def face_areas(V, F):
    validate_mesh(V, F, force_triangular=True)

    vec_ij = V[F[:,1],:] - V[F[:,0],:]
    vec_ik = V[F[:,2],:] - V[F[:,0],:]

    areas = 0.5 * np.linalg.norm(np.cross(vec_ij, vec_ik), axis=1)
    
    return areas

def vertex_areas(V, F):
    validate_mesh(V, F, force_triangular=True)
    nV = V.shape[0]

    face_area = face_areas(V, F)

    vertex_area = np.zeros(V.shape[0])
    for i in range(3):
        vertex_area += np.bincount(F[:,i], face_area, minlength=nV)
    vertex_area /= 3.
    
    return vertex_area
