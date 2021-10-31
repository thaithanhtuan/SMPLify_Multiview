import open3d as o3d
import numpy as np
from PIL import Image
import os.path as osp
import cv2
"""
Note:
{
  "Color_intrinsics" : [1280.000000, 720.000000, 605.927002, 605.799927, 639.942627, 367.209930],
  "Depth_intrinsics" : [640.000000, 576.000000, 502.724945, 502.854401, 323.970764, 326.964050]
}
(width, height, fx, fy, cx, cy)

"""
DATA_DIR = "D:/ThaiTuan/3DBODY/4.Dataset/results-0908/"

id = 0
seg_file = osp.join(DATA_DIR, "Segmentation/color_to_depth"+str(id)+".png")

im_mask = Image.open(seg_file)
mask_array = np.array(im_mask)

# parse_shape = (parse_array > 0).astype(np.float32)  # CP-VTON body shape
parse_shape = (mask_array > 0).astype(np.float32)

d2c_file = osp.join(DATA_DIR, "color2depths/depth"+str(id)+".png") # "./input/results-0908/color2depths/depth0.png"
d2c_img = cv2.imread(d2c_file, cv2.IMREAD_UNCHANGED)

depth = d2c_img * parse_shape


new_depth_file = osp.join(DATA_DIR, "new_depth/color_to_depth"+str(id)+".png")
cv2.imwrite(new_depth_file, depth.astype(np.uint16))

temp = cv2.imread(DATA_DIR + "color2depths/color_to_depth"+str(id)+".png", cv2.IMREAD_UNCHANGED)
temp = cv2.cvtColor(temp, cv2.COLOR_BGRA2BGR)
cv2.imwrite(DATA_DIR + "color2depths/color_to_depth"+str(id)+".png", temp)

color = o3d.io.read_image(DATA_DIR + "color2depths/color_to_depth"+str(id)+".png")
depth = o3d.io.read_image(new_depth_file)

rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(color, depth, convert_rgb_to_intensity=False)

intrinsic = o3d.camera.PinholeCameraIntrinsic(640, 576, 502.724945, 502.854401, 323.970764, 326.964050)
extrinsic = np.identity(4)

pcd = o3d.geometry.PointCloud.create_from_rgbd_image(image=rgbd, intrinsic = intrinsic, extrinsic = extrinsic)

o3d.visualization.draw_geometries([pcd], width=600, height=600)

o3d.io.write_point_cloud("./input/results-0908/clean_segmentation/color_to_depth"+str(id)+".ply", pcd, write_ascii = True)


