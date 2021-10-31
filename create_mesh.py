from utils import *

if __name__ == '__main__':
    create_poisson_mesh("./input/my_pcd.ply", voxel_size=5, out_dir="./input", viz=True)