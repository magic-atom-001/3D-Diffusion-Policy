import zarr
import numpy as np
from visualizer import visualize_pointcloud

# 1. 加载 zarr 数据
z = zarr.open("/home/wulong/Y/3D-Diffusion-Policy/3D-Diffusion-Policy/data/adroit_hammer_expert.zarr", mode='r')

# 2. 取一个样本的点云，例如第 6 个
pointcloud = np.array(z['data']['point_cloud']['6.0.0'])  # (512, 6)

# 3. 可视化（会自动打开浏览器窗口）
visualize_pointcloud(pointcloud)
