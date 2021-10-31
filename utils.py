import os
import open3d as o3d
import numpy as np
import cv2

def normalize_v3(arr):
    # Normalize a numpy array of 3 component vectors shape=(n,3)
    lens = np.sqrt(arr[:, 0]**2 + arr[:, 1]**2 + arr[:, 2]**2)
    arr[:, 0] /= lens
    arr[:, 1] /= lens
    arr[:, 2] /= lens
    return arr

def find2ndaxis(faces, v_normal, v_ref):

    debug  = True
    n_vertex = v_ref.shape[0] 


    # 1. first find the smallest-indexed neighbor vertex 
    # @TODO any way to speed up this step?
    ngbr_vertex =  n_vertex*np.ones(v_ref.shape[0], dtype = np.int64)
    for fidx, fv in enumerate(faces):
         v0, v1, v2 =  fv
         if ngbr_vertex[v0] > min(v1, v2): ngbr_vertex[v0] = min(v1, v2) # short-form by Matiur
         if ngbr_vertex[v1] > min(v0, v2): ngbr_vertex[v1] = min(v0, v2) 
         if ngbr_vertex[v2] > min(v1, v0): ngbr_vertex[v2] = min(v1, v0)

    # check results 
    if debug:
       for idx in range(n_vertex):
          if ngbr_vertex[idx] >= n_vertex:
              print('This vertex has no neighbor hood:',  idx)


    # 2. compute the tangential vector component 
    #    vec -   dot(normal, vec) * normal
    from numpy import dot
    from numpy.linalg import norm

    vec1 = v_ref[ngbr_vertex] - v_ref       # get the edge vector 
    # print('shape comp: ',  v_normal.shape, vec1.shape)
    coefs = np.sum(v_normal*vec1, axis=1) # coef = dot(v_normal, vec1)
    vec2 = vec1 - coefs[:, None]*v_normal  # remove the normal components

    axis = normalize_v3(vec2)

    return axis

def setup_vertex_local_coord(faces, vertices):

    # 1.1 normal vectors (1st axis) at each vertex
    _, axis_z = calc_normal_vectors(vertices, faces)
    # 1.2 get 2nd axis
    axis_x = find2ndaxis(faces, axis_z, vertices)
    # 1.3 get 3rd axis
    # matiur contribution. np.cross support row-vectorization
    axis_y = np.cross(axis_z[:, :], axis_x[:, :])

    return axis_x, axis_y, axis_z

def _remove_small_clusters(sPCD, voxel_size):

    # 4.3 take only Foot part remove small clusters  
    # 4.3.1 clustering , 파라메터 튜닝 필요!!!
    seglabels = sPCD.cluster_dbscan(eps = voxel_size*3, min_points = 20, print_progress=False)
    labelset, label_indexes, label_counts = np.unique(seglabels, return_index = True, return_counts= True)
    max_label = len(labelset)                                  
                               
    max_idx = np.argmax(label_counts)
    foot_label = labelset[max_idx]  
    
    # 4.3.2 taking the biggest cluster (foot)  
    # get only foot points and color
    # to numpy (why not using numpy Open3D?)
    points_before = np.asarray(sPCD.points)
    colors_before = np.asarray(sPCD.colors)
    # select  
    points = points_before[seglabels == foot_label, : ]      
    colors = colors_before[seglabels == foot_label, : ]  
    # reconstruct points and colors  
    #sPCD.points =  o3d.utility.Vector3dVector(points)    
    tPCD = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    tPCD.colors =  o3d.utility.Vector3dVector(colors)

    return tPCD

def remove_small_clusters(pcd, voxel_size):

    sPCD = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd.vertices))

    # 4.3 take only Foot part remove small clusters  
    # 4.3.1 clustering , 파라메터 튜닝 필요!!!
    seglabels = sPCD.cluster_dbscan(eps = voxel_size*3, min_points = 20, print_progress=False)
    labelset, label_indexes, label_counts = np.unique(seglabels, return_index = True, return_counts= True)
    max_label = len(labelset)                                  
                               
    max_idx = np.argmax(label_counts)
    foot_label = labelset[max_idx]  
    
    # 4.3.2 taking the biggest cluster (foot)  
    # get only foot points and color
    # to numpy (why not using numpy Open3D?)
    # select  
    pcd.vertices = pcd.vertices[seglabels == foot_label, : ]      
    pcd.colors = pcd.colors[seglabels == foot_label, : ]  

    return pcd

def delete_black_points(pcd):
    # 검은 색 분리
    colors = pcd.colors[:,:3] # alpha값이 있는 경우 없앰
    no_black = (colors != 0).any(axis=1)

    pcd.vertices = pcd.vertices[no_black]
    pcd.colors = pcd.colors[no_black]

# FIXME 모든 경우에 반드시 적용되는 함수가 아니라서 수정 필요
def delete_black_and_noise_for_standing_human(pcd):
    """반드시 노이즈 심한(노이즈 분산이 큰) 경우에만 사용"""
    # 검은 색 분리
    np_pcd = pcd.vertices.copy()
    np_pcdc = pcd.colors.copy()[:,:3] # alpha 값 없앰

    no_black = (np_pcdc != 0).any(axis=1)
    np_pcd = np_pcd[no_black]
    np_pcdc = np_pcdc[no_black]

    # SVD
    u, r, v = np.linalg.svd(np_pcd, full_matrices=False)
    r[1:2] = 0 # 사람의 키 방향으로 찌그러트림
    temp = np.matmul(np.matmul(u, np.diag(r)), v)

    dist = np.sum((temp - np.mean(temp, axis=0))**2, axis=1)**0.5
    dist_median = np.median(dist)

    inliers = dist < 5 * dist_median

    pcd.vertices = pcd.vertices[no_black][inliers]
    pcd.colors = pcd.colors[no_black][inliers]

    # 군집 제거
    o3dpcd = o3d.geometry.PointCloud()
    o3dpcd.points = o3d.utility.Vector3dVector(pcd.vertices)
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(o3dpcd.cluster_dbscan(eps=20, min_points=10, print_progress=True))

    num_label = labels.max() + 1
    inlier_point_num = 0
    inlier_label = 0
    for i in range(num_label):
        point_num = np.sum(labels == i)
        if inlier_point_num < point_num:
            inlier_point_num = point_num
            inlier_label = i

    pcd.vertices = pcd.vertices[labels == inlier_label]
    pcd.colors = pcd.colors[labels == inlier_label]
    
    return pcd

def estimate_normals(vertices, radius, orient=None, kmean=0):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    
    if orient:
        pcd.orient_normals_to_align_with_direction(orient) # FIXME

    normals = np.asarray(pcd.normals)
    
    if kmean > 0:
        # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        compactness, labels, centers = cv2.kmeans(vertices.astype(np.float32), kmean, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        displacement = vertices - centers[labels.flatten()]
        inner = np.sum(normals * displacement, axis=1)
        normals[inner < 0] *= -1    

    return normals

def calc_normal_vectors(vertices, faces):

    # Create a zero array with the same type and shape as our vertices i.e., per vertex normal
    _norm = np.zeros(vertices.shape, dtype=vertices.dtype)
    # Create an indexed view into the vertex array using the array of three indices for triangles
    tris = vertices[faces]
    # Calculate the normal for all the triangles, by taking the cross product of the vectors v1-v0, and v2-v0 in each triangle
    n = np.cross(tris[::, 1] - tris[::, 0], tris[::, 2] - tris[::, 0])
    # n is now an array of normals per triangle. The length of each normal is dependent the vertices,
    # we need to normalize these, so that our next step weights each normal equally.
    normalize_v3(n)
    #n = norm(n)
    # now we have a normalized array of normals, one per triangle, i.e., per triangle normals.
    # But instead of one per triangle (i.e., flat shading), we add to each vertex in that triangle,
    # the triangles' normal. Multiple triangles would then contribute to every vertex, so we need to normalize again afterwards.
    # The cool part, we can actually add the normals through an indexed view of our (zeroed) per vertex normal array
    _norm[faces[:, 0]] += n
    _norm[faces[:, 1]] += n
    _norm[faces[:, 2]] += n
    normalize_v3(_norm)
    # norm(_norm)

    return n, _norm

def create_poisson_mesh(plyFilePath, voxel_size, out_dir=None, viz=False):

    pcd = o3d.io.read_point_cloud(plyFilePath)
    pcd = _remove_small_clusters(pcd, voxel_size)

    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2*voxel_size, max_nn=30))

    pcd_points = np.asarray(pcd.points)
    pcd_normals = np.asarray(pcd.normals)

    # k-mean 알고리즘 사용하여 노말벡터 방향 지정
    num_label = 70
    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    compactness, labels, centers = cv2.kmeans(pcd_points.astype(np.float32), num_label, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    displacement = pcd_points - centers[labels.flatten()]
    inner = np.sum(pcd_normals * displacement, axis=1)
    pcd_normals[inner < 0] *= -1

    pcd.normals = o3d.utility.Vector3dVector(pcd_normals)

    if viz:
        o3d.visualization.draw_geometries([pcd], point_show_normal=True)

    with o3d.utility.VerbosityContextManager(
        o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9)
            
    if viz:
        o3d.visualization.draw_geometries([mesh])
    
    if out_dir:
        o3d.io.write_triangle_mesh(os.path.join(out_dir, "mesh.obj"), mesh)

    return mesh