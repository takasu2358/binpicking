import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pandas as pd
import cv2
import time
import pathlib
import os
import sys 

current_folder_path = pathlib.Path(__file__).resolve().parent
data_folder_path = str((current_folder_path / ".." / "data_folder").resolve())

# num = 3
# WINDOW_WIDTH = 420
# WINDOW_HEIGHT = 360
# MAX_DEPTH = 1300
# MIN_DEPTH = 800
# ROTATE180 = 1
# left = -90
# right = WINDOW_WIDTH + left
# up = 190
# bottom = up - WINDOW_HEIGHT
# high = 1300
# low = 800

num = 3
WINDOW_WIDTH = 490
WINDOW_HEIGHT = 420
left = -200
right = WINDOW_WIDTH + left
up = 190
bottom = up - WINDOW_HEIGHT
high = 1300
low = 800

def pointcloud_process(pcd_matrix):
    pcd_matrix2 = pcd_matrix[(pcd_matrix[:,0]>left)&(pcd_matrix[:,0]< right) & (pcd_matrix[:, 1]>bottom)&(pcd_matrix[:, 1]<up) & (pcd_matrix[:, 2]>low) & (pcd_matrix[:, 2]<high)]
    pcd_matrix2[:, 0] += -left
    pcd_matrix2[:, 1] += -bottom

    depth = xyz2matrix(pcd_matrix2)

    return depth

def PtoD(xyz):
    depth = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH))
    x = [[] for i in range(0, len(xyz))]
    y = [[] for i in range(0, len(xyz))]
    z = [[] for i in range(0, len(xyz))]

    for i in range(0, len(xyz)):
        x[i] = int(xyz[i][1])
        y[i] = int(xyz[i][0])
        z[i] = xyz[i][2]

    for i in range(0, len(xyz)):
        if z[i] < MIN_DEPTH:
            depth[x[i]][y[i]] = 255
        if z[i] > MAX_DEPTH:
            depth[x[i]][y[i]] = 0
        if z[i] >= MIN_DEPTH and z[i] <= MAX_DEPTH:
            pixel = 255 * (1.0 - (float(z[i] - MIN_DEPTH) / float(MAX_DEPTH - MIN_DEPTH)))
            depth[x[i]][y[i]] = pixel
    
    return depth

def xyz2matrix(matrix):
    depth = np.zeros((WINDOW_HEIGHT*num, WINDOW_WIDTH*num))
    for xyz in matrix:
        x = int(xyz[1]*num)
        y = int(xyz[0]*num)
        depth[x][y] = high - xyz[2]
    depth = max_pooling(depth)
    return depth

def max_pooling(depth):
    small_depth = np.zeros((WINDOW_HEIGHT, WINDOW_WIDTH))
    for x in range(WINDOW_HEIGHT):
        for y in range(WINDOW_WIDTH):
            kernel = depth[x*3:x*3+3, y*3:y*3+3]
            max_num = np.max(kernel)
            small_depth[x][y] = max_num
    
    return small_depth


def main(filepath):
    pcd = o3d.io.read_point_cloud(filepath)
    pcd_matrix = np.asanyarray(pcd.points)

    depth = pointcloud_process(pcd_matrix)

    # depth = PtoD(pcd_matrix2)

    # depth = cv2.rotate(depth, cv2.ROTATE_180)

    # plt.imshow(depth)
    # plt.show()

    return depth

if __name__ == "__main__":
    start = time.time()
    try:
        mode = int(sys.argv[1])
    except IndexError:
        mode = 0   

    if mode:
        print("Photograph PointCloud by PhoXi")
        root_dir = os.environ['HOME']
        robot_dir = os.path.join(root_dir, "bin-picking-robot")
        ply_path = os.path.join(data_folder_path, "ply/no-obj.ply")
        sys.path.append(str(robot_dir))      
        from bpbot.driver.phoxi import phoxi_client as pclt
        pxc = pclt.PhxClient(host="127.0.0.1:18300")
        pxc.triggerframe()
        pc = pxc.getpcd()
        pxc.saveply(ply_path) 

    filepath = os.path.join(data_folder_path, "ply/2022-11-23/no-obj.ply")
    save_png_path = os.path.join(data_folder_path, "result/flat.png")
    save_csv_path = os.path.join(data_folder_path, "csv/flat.csv")
    depth = main(filepath)
    df  = pd.DataFrame(depth)
    df.to_csv(save_csv_path, header=False, index=False)
    
    cv2.imwrite(save_png_path, depth)

    elapsed_time = time.time() - start#処理の終了時間を取得
    print("実行時間は{}秒でした．".format(elapsed_time))

    plt.imshow(depth)
    plt.show()
