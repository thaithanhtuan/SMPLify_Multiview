import open3d as o3d 
import numpy as np 
import matplotlib.pyplot as plt



def cluster_Dbscan(pcd, min_points):
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(
            pcd.cluster_dbscan(eps=20, min_points = min_points, print_progress=True))

    #max_label = labels.max()

    labelset, label_index, label_count = np.unique(labels, return_index=True, return_counts=True)
    labelset = labelset[1:]
    label_index = label_index[1:]
    label_count = label_count[1:]

    max_label = len(labelset)

    max_idx = np.argmax(label_count)

    body_label = labelset[max_idx]

    pointsBefore = np.asarray(pcd.points)
    colorsBefore = np.asarray(pcd.colors)

    points = pointsBefore[labels == body_label, :]
    colors = colorsBefore[labels == body_label, :]

    rPly = o3d.geometry.PointCloud()
    rPly.points = o3d.utility.Vector3dVector(points)
    rPly.colors = o3d.utility.Vector3dVector(colors)
    
    return rPly
import time

if __name__ == "__main__":
    
    mainPly = o3d.io.read_point_cloud("./input/results-0908/color2depths/color_to_depth7.ply")

    time1 = time.time()
    o3d.visualization.draw_geometries([mainPly],
                                      zoom=0.455,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])
    # 1st cluster DBscan
    clusterPly = cluster_Dbscan(mainPly, 20)
    o3d.visualization.draw_geometries([clusterPly],
                                      zoom=0.455,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])

    plane_model, inliers = clusterPly.segment_plane(distance_threshold=10,
                                            ransac_n=3,
                                            num_iterations=1000)

    outlier_cloud = clusterPly.select_by_index(inliers, invert=True)
    o3d.visualization.draw_geometries([outlier_cloud],
                                      zoom=0.455,
                                      front=[-0.4999, -0.1659, -0.8499],
                                      lookat=[2.1813, 2.0619, 2.0999],
                                      up=[0.1204, -0.9852, 0.1215])
    #  2nd cluster Dbscan
    bodyPly = cluster_Dbscan(outlier_cloud, 30)
    print(time.time() - time1)
    o3d.visualization.draw_geometries([bodyPly],
                                    zoom=0.455,
                                    front=[-0.4999, -0.1659, -0.8499],
                                    lookat=[2.1813, 2.0619, 2.0999],
                                    up=[0.1204, -0.9852, 0.1215])

    o3d.io.write_point_cloud("./input/results-0908/clean/color_to_depth7.ply", bodyPly, write_ascii = True)