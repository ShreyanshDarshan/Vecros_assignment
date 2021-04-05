import numpy as np
import open3d as o3d
import random

VOXEL_SIZE = 0.1
DIST_THRESH = 2*VOXEL_SIZE
ITERATIONS = 200

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def sample_plane(cloud):
    p1 = random.randint(0, cloud.shape[0])
    p2 = random.randint(0, cloud.shape[0])
    p3 = random.randint(0, cloud.shape[0])
    vec1 = cloud[p2] - cloud[p1]
    vec2 = cloud[p3] - cloud[p1]
    return cloud[p1], normalize(np.cross(vec1, vec2))

def consensus(cloud, pt, normal):
    diff = cloud - pt
    dot = np.matmul(diff, np.vstack(normal))
    dist = np.absolute(dot)
    inliers = dist[dist < DIST_THRESH]
    sum = inliers.shape[0]
    return sum

def get_inliers(cloud, pt, normal):
    inliers = []
    diff = cloud - pt
    dot = np.matmul(diff, np.vstack(normal))
    dist = np.absolute(dot)
    dist = np.transpose(dist)
    inliers = cloud[dist[0] < DIST_THRESH]
    outliers = cloud[dist[0] >= DIST_THRESH]
    return inliers, outliers

def get_key(e):
    return e[2]

def get_axis(p, normal):
    i_vec_1 = normalize(np.cross(normal, np.array([1, 0, 0])))
    i_vec_2 = normalize(np.cross(normal, np.array([0, 1, 0])))
    i_vec_3 = normalize(np.cross(normal, np.array([0, 0, 1])))
    if np.linalg.norm(i_vec_1) != 0:
        j_vec = np.cross(normal, i_vec_1)
        return [i_vec_1, j_vec, normal]
    elif np.linalg.norm(i_vec_2) != 0:
        j_vec = np.cross(normal, i_vec_2)
        return [i_vec_2, j_vec, normal]
    elif np.linalg.norm(i_vec_3) != 0:
        j_vec = np.cross(normal, i_vec_3)
        return np.array([i_vec_3, j_vec, normal])

def transform_cloud(cloud, axis):
    rot_mat = np.transpose(axis)
    rot_cloud = np.matmul(cloud, rot_mat)
    return rot_cloud

def get_surface_area(cloud, p, normal):
    selected_cloud, _ = get_inliers(cloud, p, normal)
    rot_cloud = transform_cloud(selected_cloud, get_axis(p, normal))
    flat_cloud = o3d.geometry.PointCloud()
    flat_cloud.points = o3d.utility.Vector3dVector(np.matmul(rot_cloud, np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])))
    flat_cloud_down = flat_cloud.voxel_down_sample(VOXEL_SIZE)
    # o3d.visualization.draw_geometries([flat_cloud_down])
    num_points = np.asarray(flat_cloud_down.points).shape[0]
    return VOXEL_SIZE*VOXEL_SIZE*num_points

print("program_started")
pcd = o3d.io.read_point_cloud("./boxes.pcd")
downpcd = pcd.voxel_down_sample(VOXEL_SIZE)
cloud = np.asarray(downpcd.points)
# o3d.visualization.draw_geometries([downpcd])

consensus_list = []
for i in range(ITERATIONS):
    p, normal = sample_plane(cloud)
    if np.linalg.norm(normal) != 0:
        c = consensus(cloud, p, normal)
        # print(i, c)
        consensus_list.append([p, normal, c])

consensus_list.sort(reverse=True, key=get_key)

inliers, outliers = get_inliers(cloud, consensus_list[0][0], consensus_list[0][1])
inlier_cloud = o3d.geometry.PointCloud()
outlier_cloud = o3d.geometry.PointCloud()
inlier_cloud.points = o3d.utility.Vector3dVector(inliers)
outlier_cloud.points = o3d.utility.Vector3dVector(outliers)

inlier_cloud.paint_uniform_color([0, 1, 0])
outlier_cloud.paint_uniform_color([1, 0, 0])

print("surface area =", get_surface_area(cloud, consensus_list[0][0], consensus_list[0][1]), "m^2")

o3d.visualization.draw_geometries([inlier_cloud + outlier_cloud])
