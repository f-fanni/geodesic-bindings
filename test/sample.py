import os, sys

import polyscope as ps
import numpy as np
import scipy 
import scipy.sparse
import scipy.sparse.linalg

# Path to where the bindings live
sys.path.append(os.path.join(os.path.dirname(__file__), "../build/"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../src/"))

import potpourri3d as pp3d

ps.init()

# Read input

## = Mesh test
V, F = pp3d.read_mesh("test/vase.ply")
#V, F = pp3d.read_mesh("/home/filippo/Desktop/masktest.obj")
#V, F = pp3d.read_mesh("test/bunny_small.ply")
ps_mesh = ps.register_surface_mesh("mesh", V, F)

# Distance
dists = pp3d.compute_distance(V, F, 4)
ps_mesh.add_scalar_quantity("dist", dists)

# Vector heat
solver = pp3d.MeshVectorHeatSolver(V, F)

# Vector heat (extend scalar)
ext = solver.extend_scalar([1, 22], [0., 6.]) 
ps_mesh.add_scalar_quantity("ext", ext)

# Vector heat (tangent frames)
basisX, basisY, basisN = solver.get_tangent_frames()
ps_mesh.add_vector_quantity("basisX", basisX)
ps_mesh.add_vector_quantity("basisY", basisY)
ps_mesh.add_vector_quantity("basisN", basisN)

# Vector heat (transport vector)
ext = solver.transport_tangent_vector(1, [6., 6.])
ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY
ps_mesh.add_vector_quantity("transport vec", ext3D)

ext = solver.transport_tangent_vectors([1, 22], [[6., 6.], [3., 4.]])
ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY
ps_mesh.add_vector_quantity("transport vec2", ext3D)

# Vector heat (log map)
logmap = solver.compute_log_map(1)
ps_mesh.add_parameterization_quantity("logmap", logmap)

""" 
# Flip geodesics
path_solver = pp3d.EdgeFlipGeodesicSolver(V,F)
for k in range(50):
    for i in range(5):
        path_pts = path_solver.find_geodesic_path(v_start=1, v_end=22+i)
        ps.register_curve_network("flip path " + str(i), path_pts, edges='line')
        
path_pts = path_solver.find_geodesic_path_poly([1173, 148, 870, 898])
ps.register_curve_network("flip path poly", path_pts, edges='line')

loop_pts = path_solver.find_geodesic_loop([1173, 148, 870, 898])
ps.register_curve_network("flip loop", loop_pts, edges='loop')

loop_pts = path_solver.find_geodesic_loop([307, 757, 190]) # this one contracts to a point
ps.register_curve_network("flip loop", loop_pts, edges='loop')


## = Point cloud test
P = V
ps_cloud = ps.register_point_cloud("cloud", P)

# == heat solver
solver = pp3d.PointCloudHeatSolver(P)

# distance
dists = solver.compute_distance(4)
dists2 = solver.compute_distance_multisource([4, 13, 784])
ps_cloud.add_scalar_quantity("dist", dists)
ps_cloud.add_scalar_quantity("dist2", dists2)

# scalar extension
ext = solver.extend_scalar([1, 22], [0., 6.])
ps_cloud.add_scalar_quantity("ext", ext)

# Vector heat (tangent frames)
basisX, basisY, basisN = solver.get_tangent_frames()
ps_cloud.add_vector_quantity("basisX", basisX)
ps_cloud.add_vector_quantity("basisY", basisY)
ps_cloud.add_vector_quantity("basisN", basisN)

# Vector heat (transport vector)
ext = solver.transport_tangent_vector(1, [6., 6.])
ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY
ps_cloud.add_vector_quantity("transport vec", ext3D)

ext = solver.transport_tangent_vectors([1, 22], [[6., 6.], [3., 4.]])
ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY
ps_cloud.add_vector_quantity("transport vec2", ext3D)

# Vector heat (log map)
logmap = solver.compute_log_map(1)
ps_cloud.add_scalar_quantity("logmapX", logmap[:,0])
ps_cloud.add_scalar_quantity("logmapY", logmap[:,1])

# Areas
vert_area = pp3d.vertex_areas(V,F)
ps_mesh.add_scalar_quantity("vert area", vert_area)
face_area = pp3d.face_areas(V,F)
ps_mesh.add_scalar_quantity("face area", face_area, defined_on='faces')


# Laplacian
L = pp3d.cotan_laplacian(V,F,denom_eps=1e-6)
M = scipy.sparse.diags(vert_area)
k_eig = 6
evals, evecs = scipy.sparse.linalg.eigsh(L, k_eig, M, sigma=1e-8)
for i in range(k_eig):
    ps_mesh.add_scalar_quantity("evec " + str(i), evecs[:,i])

# #geodesics and bezier
# path_pts = path_solver.compute_bezier_curve(v_list=[400,500,800,1000], n_rounds=8)
# ps.register_curve_network("bezier", path_pts, edges='line')
# ps.register_point_cloud("bezierp", path_pts)
# 
# bary,face_ids = path_solver.compute_bezier_curve_geopath(v_list=[400,500,800,1000], n_rounds=8)
# path_pts = np.array([ bary[i,0]*V[F[face_ids[i],0]]+bary[i,1]*V[F[face_ids[i],1]]+bary[i,2]*V[F[face_ids[i],2]] for i in range(len(face_ids))])
# ps.register_curve_network("bezier_alt", path_pts, edges='line')
# ps.register_point_cloud("bezier_altp", path_pts)

# directional field
ext = pp3d.compute_direction_field(V, F, 1)
ext = pp3d.compute_direction_field(V, F, 1)
solver = pp3d.MeshVectorHeatSolver(V, F)
# Vector heat (tangent frames)
basisX, basisY, basisN = solver.get_tangent_frames()
ext3D = ext[:,0,np.newaxis] * basisX +  ext[:,1,np.newaxis] * basisY
ps_mesh.add_vector_quantity("directional vecs", ext3D)
#ext3D = ext[:,2,np.newaxis] * basisX +  ext[:,3,np.newaxis] * basisY
#ps_mesh.add_vector_quantity("directional vecs2", ext3D)
#ext3D = ext[:,4,np.newaxis] * basisX +  ext[:,5,np.newaxis] * basisY
#ps_mesh.add_vector_quantity("directional vecs3", ext3D)
#ext3D = ext[:,6,np.newaxis] * basisX +  ext[:,7,np.newaxis] * basisY
#ps_mesh.add_vector_quantity("directional vecs4", ext3D)


import math

ListPoin = [ 29630, 2870, 16885 , 22037 ]
for poin in ListPoin:
    text="Punto"+str(poin)
    logmap = solver.compute_log_map(poin)
    distances = np.linalg.norm(logmap - [0.,50.], axis=1)
    path_pts_C=path_solver.find_geodesic_path(v_start=poin, v_end=np.argsort(distances)[0])
    ps.register_point_cloud(text ,path_pts_C, radius=0.0025)
    angle = -(math.atan2(ext[poin][1], ext[poin][0]))
    for LM in logmap:
        x = LM[0]
        y = LM[1]
        LM[0] = math.cos(angle) * (x) - math.sin(angle) * y
        LM[1] = math.sin(angle) * (x) + math.cos(angle) * y
    distances = np.linalg.norm(logmap - [0.,50.], axis=1)
    path_pts_C=path_solver.find_geodesic_path(v_start=poin, v_end=np.argsort(distances)[0])
    ps.register_point_cloud(text+"Orie" ,path_pts_C, radius=0.0025)


from timeit import default_timer as timer
solver = pp3d.TraceGeodesicsSolver(V,F)
start = timer()
for i in range(100):
    point, faceid, dir = solver.trace_geodesic(420, [10+np.random.randint(-5,5),50+np.random.randint(-5,5)])
end = timer()
print(("done in seconds: ",end - start, (end - start)/10000)) # Time in seconds, e.g. 5.38091952400282

ps.register_curve_network("straight", np.array([V[420], point]), edges='line')


#yocto beziers
yoctosolver = pp3d.YoctoMeshSolver(V,F)
baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[400,500,800,1000], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto", path_pts, edges='line')

dists3 = yoctosolver.compute_distance([4, 13, 784])
ps_cloud.add_scalar_quantity("dist3", dists3)

#given ext3D[N], basisXYN[N] check inversion
N = 420
print(ext[N])
print(ext3D[N])
print(ext[N,0,np.newaxis] * basisX[N] +  ext[N,1,np.newaxis] * basisY[N])
Mat = np.array([basisX[N],basisY[N],basisN[N],]).transpose()
print(Mat)
print(np.matmul(Mat, np.array([ 0.50548752, -0.86283392, 0])))
print('---')
print(np.linalg.solve(Mat, ext3D[N])) """

yoctosolver = pp3d.YoctoMeshSolver(V,F)
solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
dists4 = yoctosolver.compute_distance_meshpoints(barycentric_coordinates=[[0.5,0.2,0.3]], face_ids=[4200], max_distance=10000)
dists5 = solver.compute_distance_multisource_meshpoint([[0.5, 0.2, 0.3]], [4200])
ps_mesh.add_scalar_quantity("dist_yocto", dists4)
ps_mesh.add_scalar_quantity("dist_heat", dists5)

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[400,500,800,1000], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto", path_pts, edges='line')
yoctosolver.update_stored_paths(baryy, face_idsy) """

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[300,600,1200,2500], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto2", path_pts, edges='line')
print(yoctosolver.path_intersect_others(baryy, face_idsy))
yoctosolver.update_stored_paths(baryy, face_idsy) """ # intersect the prev, as intended

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[10,20,30,52], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto3", path_pts, edges='line')
print(yoctosolver.path_intersect_others(baryy, face_idsy))
yoctosolver.update_stored_paths(baryy, face_idsy) """

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[10,20,30,52], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto4", path_pts, edges='line')
print(yoctosolver.path_intersect_others(baryy, face_idsy))
yoctosolver.update_stored_paths(baryy, face_idsy) """ # completely overlapping to the prev, it says it intersects -> ok

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[52,30,20,10], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto4_2", path_pts, edges='line')
print(yoctosolver.path_intersect_others(baryy, face_idsy))
yoctosolver.update_stored_paths(baryy, face_idsy) """ # as the prev, but reversed, still intersects ok

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[10,200,230,252], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto5", path_pts, edges='line')
print(yoctosolver.path_intersect_others(baryy, face_idsy))
yoctosolver.update_stored_paths(baryy, face_idsy) """ # start point shared with another, it says it does NOT intersect -> ok

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[252, 230, 200, 10], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto6", path_pts, edges='line')
print(yoctosolver.path_intersect_others(baryy, face_idsy))
yoctosolver.update_stored_paths(baryy, face_idsy) """ # end in the beginning of another one, does not intersect

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[252, 230, 200, 52], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto7", path_pts, edges='line')
print(yoctosolver.path_intersect_others(baryy, face_idsy))
yoctosolver.update_stored_paths(baryy, face_idsy) """ # end in the end of another one, does not intersect

""" baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=[252, 230, 200, 10], subdivisions=4)
path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
ps.register_curve_network("bezier_yocto8", path_pts, edges='line')
print(yoctosolver.path_intersect_others(baryy, face_idsy))
yoctosolver.update_stored_paths(baryy, face_idsy) """ # end in the star of another one, does not intersect

num_iter = 200
while True:
    if num_iter == 0: break
    start = np.random.choice(F.shape[0])
    v_list = [start, start +9, start +16, start + 21]
    baryy,face_idsy = yoctosolver.compute_bezier_curve_vertices(v_list=v_list, subdivisions=4)
    if not yoctosolver.path_intersect_others(baryy, face_idsy):
        yoctosolver.update_stored_paths(baryy, face_idsy)
        path_pts = np.array([ baryy[i,0]*V[F[face_idsy[i],0]]+baryy[i,1]*V[F[face_idsy[i],1]]+baryy[i,2]*V[F[face_idsy[i],2]] for i in range(len(face_idsy))])
        ps.register_curve_network(f"bezier_yocto_{num_iter}", path_pts, edges='line')
        num_iter -= 1

ps.show()
