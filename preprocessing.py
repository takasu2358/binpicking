from calendar import firstweekday
from distutils.command import check
from ftplib import all_errors
from heapq import merge
from pickletools import uint8
from re import template
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random
from pyrsistent import v
from scipy.fft import skip_backend
from skimage.morphology import skeletonize
import time
import math
from PIL import Image
from scipy.signal import argrelmax
from itertools import chain
from collections import Counter
import itertools
import sys
import warnings
import csv
from datetime import datetime as dt
import os
from config import BinConfig
import pathlib
warnings.simplefilter('ignore')
sys.setrecursionlimit(10000)
sys.dont_write_bytecode = True

#パスの取得
current_folder_path = pathlib.Path(__file__).resolve().parent
data_folder_path = (current_folder_path / ".." / "data_folder").resolve()

#データ保存用フォルダの作成
tdatetime = dt.now()
tstr = tdatetime.strftime("%Y-%m-%d-%H-%M-%S")
foldername = "Experiment-" + tstr
save_folder_path = data_folder_path / "result" / "Experiment" / foldername
os.mkdir(save_folder_path)

# ply_filepath = data_folder_path / "ply" / "2022-9-1" / "ko-shape" / "ko-one-object.ply"
ply_filepath = data_folder_path / "ply" / "2022-9-1" / "U-shape" / "U-one-object.ply"

# config_path = data_folder_path / "cfg" /"config_file_wireharness.yaml"
config_path = data_folder_path / "cfg" /"config_file_Ushape.yaml"
# config_path = data_folder_path / "cfg" /"config_file_koshape.yaml"
bincfg = BinConfig(config_path)
cfg = bincfg.config

height, width, img_copy, max_depth = 0, 0, 0, 0

def get_neighbor(poi, gray_img):
    neighbor = []
    x0, y0 = poi[0], poi[1] #注目点の座標

    #近傍点取得
    for i in range(-1, 2):
        for j in range(-1, 2):
            #poi(注目点)の座標は格納しない
            if (i, j) == (0, 0):
                continue
            x, y = x0 + i, y0 + j #近傍点の座標
            #近傍点の座標が画像サイズ内かつ画素値が0より大きい
            try:
                if gray_img[x][y] > 0: 
                    neighbor.append([x, y])#neighborに格納
            except IndexError:
                continue

    return neighbor

def gray2color(gray_img):
    height, width = gray_img.shape
    color_img = np.zeros((height, width, 3)) #色情報を3次元にして作成
    for i in range(0, height):
        for j in range(0, width):
            luminosity = gray_img[i][j]
            color_img[i][j] = [luminosity, luminosity, luminosity]

    return color_img

def get_unique_list(seq):
    seen = []
    return [x for x in seq if not x in seen and not seen.append(x)]

def min_max_x(x):
    x = np.array(x)
    max_x = x.max(keepdims=True)
    min_x = x.min(keepdims=True)
    min_max_x = (x - min_x) / (max_x - min_x)
    return min_max_x

class Detect():

    def detect_region(self, image, area_threshold):
        large_area = []
        _, labels, stats, _ = cv2.connectedComponentsWithStats(np.uint8(image)) #ラベリング
        large_area_labels = np.where(stats[:, 4] > area_threshold)[0] #閾値以上のラベルを取得

        #取得したラベルの領域を取り出す
        for i in range(1, len(large_area_labels)):
            large_area.append(list(zip(*np.where(labels == large_area_labels[i]))))
        
        return large_area

    def detect_line(self, image):
        line_list = []
        image = image.astype(np.uint8)
        retval, labels = cv2.connectedComponents(image)
        for i in range(1, retval):
            line = list(map(list, (zip(*np.where(labels == i)))))
            line_list.append(line)
        
        return line_list

    def detect_singularity(self, skeleton):
        branch_point, end_point, ad_branch_point = [], [], [] #1点連結分岐点、終点、3点連結分岐点を格納する配列
        branch_img = np.zeros((height, width), dtype = np.uint8)
        skeleton[skeleton>0] = 1 #値を1に統一
        nozeros = list(zip(*np.where(skeleton > 0))) #値が1の座標を探索
        #値が1となる座標の近傍9点の値を足し合わせる
        for xy in nozeros:
            point = np.sum(skeleton[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]) #近傍3×3画素の範囲で画素値の合計を求める
            #値が4以上なら分岐点
            if point >= 4:
                branch_point.append(xy) #branch_pointに中心座標xyを格納
                branch_img[xy[0]][xy[1]] = 1 #分岐点のみで構成された画像を作成
            #値が2なら終点
            if point == 2:
                end_point.append(xy) #end_pointに終点座標xyを格納
        
        #1点連結分岐点か3点連結分岐点かを判断する
        count = 0
        branch_copy = branch_point.copy() #branch_pointをコピー

        #branch_copyに含まれる各座標に対してループ
        for xy in branch_copy:
            points = []
            #xyがbranch_pointに含まれていなければ飛ばす
            if not xy in branch_point:
                continue
            point = branch_img[xy[0]-1][xy[1]] + branch_img[xy[0]+1][xy[1]] + branch_img[xy[0]][xy[1]-1] + branch_img[xy[0]][xy[1]+1] #注目座標の上下左右の点を足し合わせる
            points.append(branch_img[xy[0]+1][xy[1]] + branch_img[xy[0]][xy[1]+1]) #注目座標の右・上の点を足し合わせる
            points.append(branch_img[xy[0]][xy[1]+1] + branch_img[xy[0]-1][xy[1]]) #注目座標の上・左の点を足し合わせる
            points.append(branch_img[xy[0]-1][xy[1]] + branch_img[xy[0]][xy[1]-1]) #注目座標の左・下の点を足し合わせる
            points.append(branch_img[xy[0]][xy[1]-1] + branch_img[xy[0]+1][xy[1]]) #注目座標の下・右の点を足し合わせる
            #pointsに2が含まれるかpointが4の場合
            if 2 in points or point == 4:
                ad_branch_point.append([]) #ad_branch_pointに3点格納するための空の配列を用意
                ad_branch_point[count].append(xy) #まず注目座標を格納
                branch_point.remove(xy) #branch_pointから座標xyを削除
                branch_img[xy[0]][xy[1]] = 0 #branch_imgから座標xyを削除
                #注目点の近傍9点を探索
                for i in range(xy[0]-1, xy[0]+2):
                    for j in range(xy[1]-1, xy[1]+2):
                        #x座標またはy座標が注目座標と一致する場合
                        if i == xy[0] or j == xy[1]: 
                            #branch_imgにおける該当座標の値が1の場合
                            if branch_img[i][j] == 1:
                                ad_branch_point[count].append((i, j)) #ad_branch_pointに格納
                                branch_point.remove((i, j)) #branch_pointから削除
                                branch_img[i][j] = 0 #branch_imgから削除
                count += 1
        
        return ad_branch_point, branch_point, end_point         

class MakeImage():

    def make_image(self, point_list, value):
        image = np.zeros((height, width), dtype = np.uint8)
        
        if value <= 0:
            for xy in point_list:
                image[xy[0]][xy[1]] = img_copy[xy[0]][xy[1]]
        else:
            for xy in point_list:
                image[xy[0]][xy[1]] = value

        return image

    def make_colorimage(self, point_list, value):
        image = np.zeros((height, width, 3), dtype = np.uint8)
        
        color = [value, value, value]
        for xy in point_list:
            image[xy[0]][xy[1]] = color

        return image
    
    def make_mini_image(self, point_list, value, margin = 30):
        point_array = np.array(point_list)
        x_list = point_array[:, 0]
        y_list = point_array[:, 1]
        minX, maxX = x_list.min(), x_list.max()
        minY, maxY = y_list.min(), y_list.max()
        miniHeight = maxX - minX + margin*2
        miniWidth = maxY - minY + margin*2
        miniImg = np.zeros((miniHeight, miniWidth))

        x_list -= minX - margin
        y_list -= minY - margin
        for i in range(0, len(point_list)):
            miniImg[x_list[i], y_list[i]] = value

        return miniImg, minX-margin, minY-margin, miniHeight, miniWidth, margin

class Visualize():
    
    def visualize_region(self, region_list):
        color_img = np.zeros((height, width, 3))
        for region in region_list:
            blue = random.random() #青色を0〜1の中でランダムに設定
            green = random.random() #緑色を0〜1の中でランダムに設定
            red = random.random() #赤色を0〜1の中でランダムに設定
            for xy in region:
                color_img[xy[0]][xy[1]] = [blue, green, red] #各領域ごとに異なる色を指定

        cv2.imshow('image', color_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return color_img

    def visualize_1region(self, region):
        MI = MakeImage()
        rimg = MI.make_image(region, 1)
        plt.imshow(rimg)
        plt.show()

    def visualize_1img(self, img1):
        plt.imshow(img1)
        plt.show()

    def visualize_2img(self, img1, img2):
        fig, axes = plt.subplots(1, 2, figsize = (20, 6))#画像を横に3つ表示
        ax = axes.ravel()
        ax[0].imshow(img1, cmap=plt.cm.gray)
        ax[0].axis('off')#軸無し
        ax[0].set_title('original')

        ax[1].imshow(img2, cmap=plt.cm.gray)
        ax[1].axis('off')#軸無し
        ax[1].set_title('closed_img')

        fig.tight_layout()#余白が少なくなるように表示
        plt.show()

    def visualize_3img(self, img1, img2, img3):
        fig, axes = plt.subplots(1, 3, figsize = (20, 6))#画像を横に3つ表示
        ax = axes.ravel()
        ax[0].imshow(img1, cmap=plt.cm.gray)
        ax[0].axis('off')#軸無し
        ax[0].set_title('original')

        ax[1].imshow(img2, cmap=plt.cm.gray)
        ax[1].axis('off')#軸無し
        ax[1].set_title('closed_img')

        ax[2].imshow(img3, cmap=plt.cm.gray)
        ax[2].axis('off')#軸無し
        ax[2].set_title('skeleton')

        fig.tight_layout()#余白が少なくなるように表示
        plt.show()

    def visualize_branch_point(self, region_skel, branch_point, ad_branch_point):
        ske = region_skel.copy()
        ske = gray2color(ske)
        for xy in branch_point:
            ske[xy[0]][xy[1]] = (1, 0, 0)
        for xys in ad_branch_point:
            for xy in xys:
                ske[xy[0]][xy[1]] = (0,0,1)

        self.visualize_1img(ske)

    def visualize_arrow(self,line_list, length):
        color = np.zeros((height, width, 3))
        MI = MakeImage()
        V = Visualize()
        for line in line_list:
            point11 = np.array(line[0])
            point12 = np.array(line[length//10])
            point13 = np.array(line[length//5])
            v1 = ((point11 - point12) + (point12 - point13))//2
            theta1 = int(math.degrees(math.atan2(v1[1], v1[0])))
            point21 = np.array(line[-1])
            point22 = np.array(line[-length//10])
            point23 = np.array(line[-length//5])
            v2 = ((point21 - point22) + (point22 - point23))//2
            theta2 = int(math.degrees(math.atan2(v2[1], v2[0])))   
            limg = MI.make_image(line, 1)
            color[limg>0] = [255, 255, 255]
            color = cv2.arrowedLine(color, [point21[1], point21[0]], [point21[1] + v2[1]*3, point21[0] + v2[0]*3], (0, 255, 0), 3)
            color = cv2.arrowedLine(color, [point11[1], point11[0]], [point11[1] + v1[1]*3, point11[0] + v1[0]*3], (255, 0, 0), 3)

class RegionGrowing():

    def __init__(self, image):
        self.img = image
        self.img_copy = img_copy
        self.height = height
        self.width = width
        if max_depth > 75:
            self.ec = cfg["extend_condition_large_depth"]
        else:
            self.ec = cfg["extend_condition_small_depth"]
        self.lat = cfg["large_area_threshold"]

    def search_seed(self):
        MI = MakeImage()
        D = Detect()

        if not self.img.ndim == 2:
            raise ValueError("入力画像は2次元(グレースケール)にしてください")

        region_list = []
        value, seed = self.serch_nonzero(0, 0)
        length = 0
        while value:
            self.img[seed[0]][seed[1]] = 0
            region = self.region_growing([seed], [])
            region.insert(0, seed)
            value, seed = self.serch_nonzero(seed[0], seed[1])
            if len(region) > self.lat:
                region_list.append(region)
                length += len(region)
                # V = Visualize()
                # V.visualize_1region(region)
        print(length)
        return region_list

    def region_growing(self, prepartial_region, region):
        if prepartial_region == []:
            return region

        for poi in prepartial_region:
            neighbor = get_neighbor(poi, self.img)
            if len(neighbor) == 0:
                continue
            partial_region = self.compare_luminosity(neighbor, poi)
            region.extend(partial_region)
            region = self.region_growing(partial_region, region)

        return region

    def compare_luminosity(self, neighbor, poi):
        partial_region = []
        poi_luminosity = self.img_copy[poi[0]][poi[1]]
        for xy in neighbor:
            neighbor_luminosity = self.img_copy[xy[0]][xy[1]]
            if np.abs(poi_luminosity - neighbor_luminosity) < self.ec:
                partial_region.append(xy)
                self.img[xy[0]][xy[1]] = 0
        
        return partial_region

    def serch_nonzero(self, x, y):
        for i in range(x, height):
            if i == x:
                for j in range(y, width):
                    value = self.img[i][j]
                    if value != 0:
                        return value, [i, j] 
            else:
                for j in range(0, width):
                    value = self.img[i][j]
                    if value != 0:
                        return value, [i, j]
        return 0, []

class CenterPoint():

    def __init__(self):
        self.Size = 51
        self.Step = 16
        self.half_S = self.Size//2
        self.nextPdist = 6 #Don't change(loop process will go wrong)

    def search_center_points(self, region_list):
        line_templates = [[] for i in range(0, self.Step)]
        center_points_list = []
        MI = MakeImage()

        st = time.time()
        ##########################テンプレートの作成############################################
        ground_img = np.zeros((self.Size, self.Size)) #テンプレートの下地を作成
        for i in range(0, self.Step):
            line_angle = ground_img.copy()
            radian = math.radians(180/self.Step*(i+1)) #thetaは各方位の角度
            y = int(self.Size*math.sin(radian)) #方位からy座標を取得
            x = int(self.Size*math.cos(radian)) #方位からx座標を取得
            line_angle = cv2.line(line_angle, (-x+self.half_S, -y+self.half_S), (x+self.half_S, y+self.half_S), 1) #直線を引く
            line_templates[i] = line_angle #画像を格納
        #######################################################################################

        #region_listに格納された各領域に対して中央線を求める
        for region in region_list:
            center_points, fore_points, back_points = [], [], []
            reg_img, minx, miny, Height, Width, _ = MI.make_mini_image(region, 1)  #領域の画像を作成
            
            #######################最初の中心点を求める#########################################
            reg_img = cv2.morphologyEx(reg_img, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))  #クロージング処理
            blur = cv2.GaussianBlur(reg_img, (45, 45), 0)  #ガウシアンブラーをかける
            bm = np.argmax(blur)  #輝度値が最大のインデックスを求め得る
            center_x, center_y = divmod(bm, Width)
            ####################################################################################
    
            pil_img = Image.fromarray(reg_img) #cropを使うためpillow型に変更
            pre_index = -1 #はじめはpre_indexは無いため、-1を代入
            center1, center2, nextPdist1, nextPdist2, pre_index, first_dist, pmsign1, pmsign2 = self.center_orientation([center_x, center_y], pre_index, 1, 0, pil_img, line_templates)
            pre_index1 = pre_index
            pre_index2 = pre_index
            shift1, shift2 = 0, 0

            #はじめの中心点からは２方向に伸びるのでwhileも２つ
            #前方向の探索
            fore_count, fore_loop_flag = 0, 0
            while 1:
                #終了条件：次の候補点の画素値が０の場合
                if reg_img[center1[0]][center1[1]] == 0:
                    break
                #点数が100点以上の場合は、ループのチェックが入る
                if fore_count > 100:
                    fore_loop_flag, _ = self.check_loop(fore_points,  -1) #最後の点に対して近傍点を確認する
                    break
                center1, current_center, current_nextPdist, pre_index1, pmsign1, dist, current_shift = self.center_orientation(center1, pre_index1, pmsign1, shift1, pil_img, line_templates)
                if pre_index1 == -100:
                    break
                fore_points.append([current_center[0]+minx, current_center[1]+miny, nextPdist1, dist, shift1])
                nextPdist1 = current_nextPdist
                shift1 = current_shift
                fore_count += 1

            #ループが存在すると判定された場合、最初の点の近傍を確認する(最初の点の超近傍に点が存在すればその点をループの終点とする)
            if fore_loop_flag == 1:
                # print("Loop exists in fore center points process")
                start_flag, min_dist_index = self.check_loop(fore_points, 0)
                # if start_flag == 0:
                #     # raise ValueError("ループ処理に場合分けが必要です")
                #     # print("ループ処理に場合分けが必要です")
                # else:
                if start_flag != 0:
                    fore_points = np.delete(fore_points, np.s_[min_dist_index:-1], 0).astype(int)

            #後ろ方向の探索
            back_count, back_loop_flag = 0, 0
            while 1:
                #終了条件：次の候補点の画素値が０の場合
                if reg_img[center2[0]][center2[1]] == 0:
                    break
                #点数が100点以上の場合は、ループのチェックが入る
                if back_count > 100:
                    back_loop_flag, _ = self.check_loop(back_points,  -1) #最後の点に対して近傍点を確認する
                    break
                center2, current_center, current_nextPdist, pre_index2, pmsign2, dist, current_shift = self.center_orientation(center2, pre_index2, pmsign2, shift2, pil_img, line_templates)
                if pre_index2 == -100:
                    break
                back_points.append([current_center[0]+minx, current_center[1]+miny, nextPdist2, dist, shift2])
                nextPdist2 = current_nextPdist
                shift2 = current_shift
                back_count += 1

            if back_loop_flag == 1:
                # print("Loop exists in back center points process")
                start_flag, min_dist_index = self.check_loop(back_points, 0)
                # if start_flag == 0:
                #     # raise ValueError("ループ処理に場合分けが必要です")
                #     print("ループ処理に場合分けが必要です")
                # else:
                if start_flag != 0:
                    back_points = np.delete(back_points, np.s_[min_dist_index:-1], 0).astype(int)

            fore_points = list(reversed(fore_points))
            center_points.extend(fore_points)
            center_points.append([center_x+minx, center_y+miny, 30, first_dist, 0])
            center_points.extend(back_points)  

            if len(center_points) > 6:
                if center_points[0][4] == 1 and center_points[2][4] == 0:
                    del center_points[0]
                if center_points[-1][4] == 1 and center_points[-3][4] == 0:
                    del center_points[-1]

            if len(center_points) > 3:
                if center_points[0][2] < 10:
                    del center_points[0]
                if center_points[-1][2] < 10:
                    del center_points[-1]

            if len(center_points) >= 2:
                dist1 = center_points[0][3]
                dist2 = center_points[1][3]
                if dist1 < dist2*0.75 and len(center_points) >= 3:
                    xy1 = [center_points[0][0], center_points[0][1]]
                    xy2 = [center_points[1][0], center_points[1][1]]
                    xy3 = [center_points[2][0], center_points[2][1]]
                    vectorx21 = xy1[0] - xy2[0]
                    vectory21 = xy1[1] - xy2[1]
                    vectorx32 = xy2[0] - xy3[0]
                    vectory32 = xy2[1] - xy3[1]
                    theta21 = int(math.degrees(math.atan2(vectory21, vectorx21)))
                    theta32 = int(math.degrees(math.atan2(vectory32, vectorx32)))
                    dif_theta321 = np.abs(theta21 - theta32)
                    if dif_theta321 > 180:
                        dif_theta321 = 360 - dif_theta321
                    if dif_theta321 > 30:
                        del center_points[0]           
                dist1 = center_points[-1][3]
                dist2 = center_points[-2][3]
                if dist1 < dist2*0.75 and len(center_points) >= 3:
                    xy1 = [center_points[-1][0], center_points[-1][1]]
                    xy2 = [center_points[-2][0], center_points[-2][1]]
                    xy3 = [center_points[-3][0], center_points[-3][1]]
                    vectorx21 = xy1[0] - xy2[0]
                    vectory21 = xy1[1] - xy2[1]
                    vectorx32 = xy2[0] - xy3[0]
                    vectory32 = xy2[1] - xy3[1]
                    theta21 = int(math.degrees(math.atan2(vectory21, vectorx21)))
                    theta32 = int(math.degrees(math.atan2(vectory32, vectorx32)))
                    dif_theta321 = np.abs(theta21 - theta32)
                    if dif_theta321 > 180:
                        dif_theta321 = 360 - dif_theta321
                    if dif_theta321 > 30:
                        del center_points[-1]

            center_points = [points[0:2] for points in center_points]
            center_points = get_unique_list(center_points)
            center_points_list.append(center_points)

        return center_points_list

    def check_loop(self, points, index):
        if not len(points[0]) == 2:
            points = np.array([row[0:2] for row in points])
        
        poi = points[index]
        points = np.delete(points, index, 0)
        dif_points = np.abs(points - poi)
        sum_points = [np.sum(point) for point in dif_points]
        sort_points = np.argsort(sum_points)
        for min_index in sort_points:
            if np.abs(min_index - index) > 5:
                break
        min_dist = sum_points[min_index]
        if  min_dist < 7:
            min_dist_index = sum_points.index(min_dist)
            if index < min_dist_index:
                min_dist_index += 1
            return 1, min_dist_index

        return 0, []

    def center_orientation(self, center_point, pre_index, pre_pmsign, shift, pil_img, line_angles):
        center_x, center_y = center_point
        cut = np.asarray(pil_img.crop((center_y-self.half_S, center_x-self.half_S, center_y+self.half_S+1, center_x+self.half_S+1))) #中心点を基準にテンプレート画像と同サイズに切り取る
        left, upper = center_x-self.half_S, center_y-self.half_S #切り取った画像の左端の座標を取得

        if pre_index == -1:
            dists = np.array([np.array([10000, 0]) for n in range(0, self.Step)]) #距離を格納する配列を作成 
            dist_ps = [[] for n in range(0, self.Step)] #直線の端の点を格納する配列を作成
        else:
            dists = np.array([np.array([10000, 0]) for n in range(0, 3)]) #距離を格納する配列を作成 
            dist_ps = [[] for n in range(0, 3)] #直線の端の点を格納する配列を作成

        dif_index_list = [0, 1, 15]
        list_index = -1
        ###############################################################################################################
        for i in range(0, self.Step):     
            dif_index = np.abs(pre_index - i)
            if dif_index in dif_index_list or pre_index == -1:
                list_index += 1
                cut_line = cut*line_angles[i]                                                       #切り出した領域にテンプレートをかけ合わせる
                line_pixel = np.nonzero(cut_line)                                                   #値を持つ座標を取得する
                if len(line_pixel[0]) == 0:                                                         #かけ合わせた画像に値がない場合次に移る
                    continue

                ########################切り取られた線の端点を求める#########################################################
                p1_x = np.min(line_pixel[0])                                                        #片方の端点のｘ座標を、x座標リストの最小値として取得する
                p1_ys = line_pixel[1][np.where(line_pixel[0] == p1_x)]                              #x座標がp1_xとなるy座標を取得
                p1_y = np.min(p1_ys)                                                                #p1_ysの中から最小のものを取り出す
                cutter = np.sum(cut_line[p1_x-1:p1_x+2, p1_y-1:p1_y+2])                             #(p1_x, p1_y)の近傍９点を足し合わせる
                if cutter > 2:                                                                      #足し合わせたものが２より大きければ、p1_yを更新
                    p1_y = np.max(p1_ys)                                                            #p1_ysの中から最大のものを取り出す

                p2_x = np.max(line_pixel[0])                                                        #もう片方の端点のx座標を、x座標のリストの最大値として取得する
                p2_ys = line_pixel[1][np.where(line_pixel[0] == p2_x)]                              #x座標がp2_xとなるy座標を取得
                p2_y = np.max(p2_ys)                                                                #p2_ysの中から最小のものを取り出す
                cutter = np.sum(cut_line[p2_x-1:p2_x+2, p2_y-1:p2_y+2])                             #(p2_x, p2_y)の近傍９点を足し合わせる
                if cutter > 2:                                                                      #足し合わせたものが２より大きければ、p2_yを更新            
                    p2_y = np.min(p2_ys)                                                            #p1_ysの中から最大のものを取り出す
                ###############################################################################################################

                ########################切り取られた線の長さを求める###########################################################
                dist = (p1_x-p2_x)**2 + (p1_y-p2_y)**2                                              #端点同士の距離を計算
                dists[list_index] = [dist, i]                                                                     #リストに距離を格納
                dist_ps[list_index] = [p1_x, p1_y, p2_x, p2_y]                                               #該当する端点座標をリストに格納
        arg_sort_dists = np.argsort(dists[:, 0])                                                          #距離を格納したリストをソートする
        
        # sort_dists = []
        # for index in arg_sort_dists:
        #     sort_dists.append(dists[index][1])
        ############################################################################################################### 

        arg_min_index = arg_sort_dists[0]
        min_index = dists[arg_min_index][1]
        dif_index = np.abs(pre_index - min_index)

        #####################min_indexとなる線分の情報取得、中心点の更新################################################
        min_x1 = dist_ps[arg_min_index][0]                                                    #最小となる線の端点のx座標を取得
        min_x2 = dist_ps[arg_min_index][2]                                                          #最小となる線の端点のx座標を取得
        min_y1 = dist_ps[arg_min_index][1]                                                          #最小となる線の端点のy座標を取得
        min_y2 = dist_ps[arg_min_index][3]                                                          #最小となる線の端点のy座標を取得
        mini_center_x = (min_x1+min_x2)//2                                                      #中心のx座標を取得した線分の中心のx座標に更新
        mini_center_y = (min_y1+min_y2)//2                                                      #中心のy座標を取得した線分の中心のy座標に更新
        min_dist = dists[arg_min_index][0]                                                             #最小となる線の長さを取得
        ################################################################################################################
        
        ##############################最初の中心点における次の候補点の取得##############################################
        if pre_index == -1:
            theta1 = math.radians(180/self.Step*min_index+90)                                   #前方向の角度
            pmsign1 = 1                                                                         #前方向の符号(theta1式中の90の前についている符号)
            xsin1 = math.sin(theta1)                                                            #前方向の次の候補点へのxベクトル
            ycos1 = math.cos(theta1)                                                            #前方向の次の候補点へのyベクトル
            next_center_xy1, return_nextPdist1, _ = self.search_next_center_point([mini_center_x, mini_center_y], xsin1, ycos1, cut, [min_x1, min_y1], [min_x2, min_y2])

            theta2 = math.radians(180/self.Step*min_index-90)
            pmsign2 = -1
            xsin2 = math.sin(theta2)
            ycos2 = math.cos(theta2)
            next_center_xy2, return_nextPdist2, _ = self.search_next_center_point([mini_center_x, mini_center_y], xsin2, ycos2, cut, [min_x1, min_y1], [min_x2, min_y2])
        
            pre_index = min_index                                                               #pre_indexに現在の方向の添字min_indexを代入
            center_x1 = next_center_xy1[0] + left                                                   #前方向の次の中心点のx座標を取得
            center_x2 = next_center_xy2[0] + left                                                   #後方向の次の中心点のx座標を取得
            center_y1 = next_center_xy1[1] + upper                                                  #前方向の次の中心点のy座標を取得
            center_y2 = next_center_xy2[1] + upper                                                  #後ろ方向の次の中心点のy座標を取得

            return [center_x1, center_y1], [center_x2, center_y2], return_nextPdist1, return_nextPdist2, pre_index, min_dist, pmsign1, pmsign2
        ################################################################################################################

        ###################################２点目以降の候補点の取得#####################################################
        if dif_index == 15:                                                                                #pre_indexとmin_indexの差が15だと90の前につく符号が入れ替わる
            pre_pmsign *= -1                                                                                #pre_pmsignの符号を入れ替える
            theta = math.radians(180/self.Step*(min_index+1)+(90*pre_pmsign))                               #次の候補点が存在する方向を取得
        else:
            theta = math.radians(180/self.Step*(min_index+1)+(90*pre_pmsign))                               #符号を入れ替えずに次の候補点が存在する方向を取得

        xsin = math.sin(theta)                                                                              #次の候補点へのxベクトル
        ycos = math.cos(theta)                                                                              #次の候補点へのyベクトル
        next_center_xy, return_nextPdist, shift = self.search_next_center_point([mini_center_x, mini_center_y], xsin, ycos, cut, [min_x1, min_y1], [min_x2, min_y2])

        if next_center_xy == [-100, -100]:
            pre_index = -100
        else:
            pre_index = min_index                                                                               #pre_indexに現在の方向の添字min_indexを代入
        current_center_x = mini_center_x + left                                                             #現在の中心点のx座標を取得
        current_center_y = mini_center_y + upper                                                            #現在の中心点のy座標を取得
        next_center_x = next_center_xy[0] + left                                                                #次の中心点のx座標を取得
        next_center_y = next_center_xy[1] + upper                                                               #次の中心点のy座標を取得

        return [next_center_x, next_center_y], [current_center_x, current_center_y], return_nextPdist, pre_index, pre_pmsign, min_dist, shift

    def search_next_center_point(self, mini_center_xy, xsin, ycos, cut, min_xy1, min_xy2):
        mini_center_x = mini_center_xy[0]
        mini_center_y = mini_center_xy[1]
        shift = 0

        nextPdist2 = self.nextPdist*(2/3)
        nextPdist3 = nextPdist2*(1/2)
        long_x = int(self.nextPdist*xsin)
        long_y = int(self.nextPdist*ycos)
        medium_x = int(nextPdist2*xsin)
        medium_y = int(nextPdist2*ycos)
        short_x = int(nextPdist3*xsin)
        short_y = int(nextPdist3*ycos)
        if short_x == 0 and short_y == 0:
            if nextPdist3*xsin >= nextPdist3*ycos:
                short_x = 1
            else:
                short_y = 1

        next_center_x = mini_center_x+long_x                                              #切り取られた画像上で、次の候補点のx座標を取得
        next_center_y = mini_center_y+long_y                                              #切り取られた画像上で、次の候補点のy座標を取得

        if cut[next_center_x][next_center_y] == 0:                                                          #次の候補点が黒い領域内なら、距離を短くしてもう一度探索                                                            #はじめの距離の３分の２に縮小
            next_center_x = mini_center_x+medium_x                                              #切り取られた画像上で、次の候補点のx座標を取得
            next_center_y = mini_center_y+medium_y                                              #切り取られた画像上で、次の候補点のy座標を取得
            if cut[next_center_x][next_center_y] == 0:                                                      #まだ黒い領域内なら、もう一度だけ距離を短くして探索                                                              #はじめの距離の３分の１に縮小
                next_center_x = mini_center_x+short_x                                          #切り取られた画像上で、次の候補点のx座標を取得
                next_center_y = mini_center_y+short_y                                          #切り取られた画像上で、次の候補点のy座標を取得
                if cut[next_center_x][next_center_y] == 0:                                                  #それでも黒い領域なら、切り取られた線上で中心点をずらして候補点探索

                    ####################################中心点をずらす############################################################################
                    shift = 1                                                                               #中心点をずらしたかのフラグ
                    cut_line = np.zeros((self.Size, self.Size))                                            
                    cut_line = cv2.line(cut_line, (min_xy1[1], min_xy1[0]), (mini_center_y, mini_center_x), 1)      #min_indexの方向の切り取られた線を画像上で作成
                    cut_line = cv2.line(cut_line, (min_xy2[1], min_xy2[0]), (mini_center_y, mini_center_x), 1)      #min_indexの方向の切り取られた線を画像上で作成
                    poi = [mini_center_x, mini_center_y]                                                    #現在の中心点を注目点に
                    next_pois = get_neighbor(poi, cut_line)                                                #近傍点探索
                    cut_line[mini_center_x][mini_center_y] = 0                                              #注目点を画像上から削除
                    if len(next_pois) == 2:                                                                 #近傍点が２点のとき
                        next_poi1 = [[next_pois[0][0], next_pois[0][1]]]                                                            #中心点の左の点を取得
                        next_poi2 = [[next_pois[1][0], next_pois[1][1]]]                                                            #中心点の右の点を取得
                        while 1:                                                                            #候補点が見つかるか、探索できる点がなくなるまで続ける
                            if len(next_poi1) > 0: 
                                next_poi1 = next_poi1[0]                          
                                mini_center_x1 = next_poi1[0]                                                   #中心点のx座標を更新
                                mini_center_y1 = next_poi1[1]                                                   #中心点のy座標を更新
                                next_center_x = mini_center_x1+long_x                         #nextPdistの距離で候補点のx座標を探索
                                next_center_y = mini_center_y1+long_y                         #nextPdistの距離で候補点のy座標を探索
                                if cut[next_center_x][next_center_y] == 0:
                                    next_center_x = mini_center_x1+medium_x                         #nextPdist2の距離で候補点のx座標を探索
                                    next_center_y = mini_center_y1+medium_y                         #nextPdist2の距離で候補点のy座標を探索
                                    if cut[next_center_x][next_center_y] == 0:
                                        next_center_x = mini_center_x1+short_x                     #nextPdist3の距離で候補点のx座標を探索
                                        next_center_y = mini_center_y1+short_y                     #nextPdist3の距離で候補点のy座標を探索
                                        if cut[next_center_x][next_center_y] == 0:                              #候補点の探索失敗
                                            next_poi1 = get_neighbor(next_poi1, cut_line)                       #現在の中心点の近傍点を探索
                                            cut_line[mini_center_x1][mini_center_y1] = 0                        #現在の中心点を画像上から削除
                                        else:                                       
                                            return_nextPdist = nextPdist3
                                            break
                                    else:
                                        return_nextPdist = nextPdist2
                                        break
                                else:
                                    return_nextPdist = self.nextPdist
                                    break
                            
                            if len(next_poi2) > 0:
                                next_poi2 = next_poi2[0]
                                mini_center_x2 = next_poi2[0]
                                mini_center_y2 = next_poi2[1]
                                next_center_x = mini_center_x2+long_x
                                next_center_y = mini_center_y2+long_y
                                if cut[next_center_x][next_center_y] == 0:
                                    next_center_x = mini_center_x2+medium_x 
                                    next_center_y = mini_center_y2+medium_y 
                                    if cut[next_center_x][next_center_y] == 0:
                                        next_center_x = mini_center_x2+short_x
                                        next_center_y = mini_center_y2+short_y
                                        if cut[next_center_x][next_center_y] == 0:
                                            next_poi2 = get_neighbor(next_poi2, cut_line)
                                            cut_line[mini_center_x2][mini_center_y2] = 0
                                        else:
                                            return_nextPdist = nextPdist3
                                            break
                                    else:
                                        return_nextPdist = nextPdist2
                                        break
                                else:
                                    return_nextPdist = self.nextPdist
                                    break
                            
                            if len(next_poi1) == 0 and len(next_poi2) == 0:
                                return_nextPdist = 0
                                break
                    else:
                        return_nextPdist = 0
                    ###############################################################################################################################

                else:
                    return_nextPdist = nextPdist3
            else:
                return_nextPdist = nextPdist2
        else:
            return_nextPdist = self.nextPdist

        # if [next_center_x, next_center_y] == [self.half_S, self.half_S]:
        #     [next_center_x, next_center_y] = [-100, -100]

        return [next_center_x, next_center_y], return_nextPdist, shift

class ConnectRegion():

    def __init__(self):
        self.full_length = cfg["full_length"] 
        self.error_length = self.full_length//5
        self.line_length_threshold = self.full_length*0.2

    def skip_by_length(self, region_list2, center_points_list):
        rregion, new_center_points_list, skip_index, line_list, line_length_list = [], [], [], [], []
        for i, (region, center_point) in enumerate(zip(region_list2, center_points_list)):
            if len(center_point) > 3:
                line = self.connect_points(center_point)
                length = len(line)
                if length > self.line_length_threshold:
                    new_center_points_list.append(center_point)
                    rregion.append(region)
                    line_list.append(line)
                    line_length_list.append(length)
                else:
                    skip_index.append(i)
            else:
                skip_index.append(i)

        # line_list = self.cut_line(line_list)
                
        return rregion, new_center_points_list, skip_index, line_list, line_length_list

    def cut_line(self, line_list):
        cut_line_list = []
        for line in line_list:
            length = len(line)
            del line[-length//10:0]
            del line[0:length//10]
            cut_line_list.append(line)
        return cut_line_list

    def connect_points(self, points):
        line_xy = []
        for i in range(len(points) - 1):
            pre_xy = points[i]
            current_xy = points[i+1]
            line_xy = self.connect_point(pre_xy, current_xy, line_xy)
        line_xy.append(current_xy[0:2])

        return line_xy

    def connect_point(self, xy, next_xy, line_xy):
        pre_x, pre_y = xy[0], xy[1]
        current_x, current_y = next_xy[0], next_xy[1]
        line_xy.append([pre_x, pre_y])
        if np.abs(pre_x - current_x) > np.abs(pre_y - current_y):
            if pre_x == current_x:
                tan = 0
            else:
                tan = (pre_y - current_y) / (pre_x - current_x)
            if pre_x <= current_x:
                for x in range(pre_x+1, current_x, 1):
                    xt = x - pre_x
                    y = int(tan * xt) + pre_y
                    line_xy.append([x, y])
            else:
                for x in range(pre_x-1, current_x, -1):
                    xt = x - pre_x
                    y = int(tan * xt) + pre_y
                    line_xy.append([x, y])
        else:
            if pre_y == current_y:
                tan = 0
                tan2 = 0
            else:
                tan = (pre_x - current_x) / (pre_y - current_y)
            if pre_y <= current_y:
                for y in range(pre_y+1, current_y, 1):
                    yt = y - pre_y
                    x = int(tan * yt) + pre_x
                    line_xy.append([x, y])
            else:
                for y in range(pre_y-1, current_y, -1):
                    yt = y - pre_y
                    x = int(tan * yt) + pre_x
                    line_xy.append([x, y])

        return line_xy

    def first_check(self, line_list, line_length_list):
        correct_connection_index = []
        first_check_result = self.check_connect_accuracy(line_length_list)

        [correct_connection_index.append([index]) for index in range(len(first_check_result)) if first_check_result[index] == 1]

        return first_check_result, correct_connection_index

    def end_point_information(self, line_list):
        thetas, ends, depthes = [], [], []
        depth_img = img_copy.copy()
        for line in line_list:
            thetas.append([[], []])                                                                        
            ends.append([[], []])
            depthes.append([])
            length = len(line)   

            point11 = np.array(line[0])
            point12 = np.array(line[length//10])
            point13 = np.array(line[length//5])
            v1 = ((point11 - point12) + (point12 - point13))//2
            theta1 = int(math.degrees(math.atan2(v1[1], v1[0])))
            depth = np.max(depth_img[point12[0]-1:point12[0]+2, point12[1]-1:point12[1]+2])
            if depth == 0:
                print("check point!!")
            thetas[-1][0] = theta1                                                                                           
            ends[-1][0] = point12
            depthes[-1].append(int(depth))

            point21 = np.array(line[-1])
            point22 = np.array(line[-length//10])
            point23 = np.array(line[-length//5])
            v2 = ((point21 - point22) + (point22 - point23))//2
            theta2 = int(math.degrees(math.atan2(v2[1], v2[0])))   
            depth = np.max(depth_img[point22[0]-1:point22[0]+2, point22[1]-1:point22[1]+2])
            if depth == 0:
                print("check point!!")                                                             
            thetas[-1][1] = theta2                                                                                           
            ends[-1][1] = point22
            depthes[-1].append(int(depth))

        V = Visualize()
        V.visualize_arrow(line_list, length)
        return thetas, ends, depthes

    def connect_region(self, center_points_list, rregion, first_check_result, line_list):
        line_num = len(line_list)                                                                                   #region_numに領域の数を代入
        combi, side, cost_list = [], [], []                                                        
        V = Visualize()
        MI = MakeImage()
        thetas, ends, depthes = self.end_point_information(line_list)

        ################################用語集######################################################
        #sum_theta : 2つの端点の向きの合計(互いに向き合っているときが最小となる)
        #dif_theta : 2つの端点を結んだ線の向きと端点の向きの差分
        #distance : 2つの端点の距離
        #dif_depth : 2つの端点の深度の差分
        ###########################################################################################

        ##################################接続条件############################################################
        sum_theta_threshold = 30                                                                                            #sum_thetaに対する閾値
        dif_theta_threshold = 50                                                                                            #dif_thetaに対する閾値
        distance_threshold = 75                                                                                             #distanceに対する閾値
        min_distance_threshold = 0                                                                                        #distanceに対する閾値
        cost_threshold = 1.7

        #thetas,ends,center_point_depthの情報からsum_theta,dif_theta,distance,dif_depthを求め、コストを計算することで接続の有無を取得する
        for i in range(0, line_num-1):                                                                                    #対象となる1つ目の領域番号
            for j in (0, 1):                                                                                                #該当する領域の端点番号(端点は1つの領域に2つ)
                if first_check_result[i] == 1:
                    continue
                theta1 = thetas[i][j]                                                                                       
                end1 = ends[i][j]
                depth1 = depthes[i][j]
                for m in range(i+1, line_num):                                                                            #対象となる2つ目の領域番号(全領域、端点を総当りで調べる)
                    for n in (0, 1):                                                                                        #該当する領域の端点番号(端点は1つの領域に2つ)
                        if first_check_result[m] == 1:
                            continue
                        theta2 = thetas[m][n]
                        end2 = ends[m][n]
                        depth2 = depthes[m][n]
                        
                        sum_theta = np.abs(np.abs(theta1 - theta2) - 180)
                        sum_theta /= 180

                        distance = int(np.sqrt((end1[0]-end2[0])**2 + (end1[1]-end2[1])**2))                                #distanceの計算
                        if distance > 150:
                            continue
                        cost_distance = distance / self.full_length

                        #dif_thetaの計算
                        vx12 = end2[0] - end1[0]                                                                            #端点同士を結んだ線のx軸の向き
                        vy12 = end2[1] - end1[1]                                                                            #端点同士を結んだ線のy軸の向き
                        theta12 = int(math.degrees(math.atan2(vy12, vx12)))                                                 #端点1から見た端点同士を結んだ線の向きを計算
                        dif_theta1 = np.abs(theta12 - theta1)                                                               #端点1の向きと端点の同士を結んだ線の向きとの差分を求める
                        if dif_theta1 > 180:                                                                                #dif_tetha1を0~180に収める
                            dif_theta1  = 360 - dif_theta1
                        theta21 = int(math.degrees(math.atan2(-vy12, -vx12)))                                               #端点2から見た端点同士を結んだ線の向きを計算
                        dif_theta2 = np.abs(theta21 - theta2)                                                               #端点2の向きと端点の同士を結んだ線の向きとの差分を求める
                        if dif_theta2 > 180:                                                                                #dif_theta2を0~180に収める
                            dif_theta2 = 360 - dif_theta2
                        dif_theta = (dif_theta1 + dif_theta2)/360                                                            #dif_theta1とdif_theta2の平均を計算                                  

                        dif_depth = np.abs(depth1 - depth2) 
                        dif_depth /= 255

                        #costを計算(distanceが近すぎるとdif_depthが望ましくない結果を出すため、distanceに閾値を定める)
                        # if distance <= min_distance_threshold:
                        #     cost = cost_distance + sum_theta + dif_theta + dif_depth                                #distanceが閾値より小さければ、dif_depthを3分の1にする
                        # else:
                        cost = cost_distance + sum_theta//40 + dif_theta*3 + dif_depth*3

                        print(distance, cost_distance, sum_theta, dif_theta, dif_depth, cost)
                        # self.checkWhereConnected(line_list, i, j, m, n)

                        #costが閾値より小さければ接続の対象とする
                        if cost < cost_threshold:
                            # self.checkWhereConnected(line_list, i, j, m, n)
                            
                            flag = True

                            #得られた接続先がすでに格納されているかチェック
                            dupis = np.where(np.array(combi).flatten() == i)[0]
                            for dupi in dupis:
                                q, mod = divmod(dupi, 2)
                                if j == side[q][mod]:
                                    if cost < cost_list[q]:
                                        flag = True
                                        del combi[q]
                                        del side[q]
                                        del cost_list[q]
                                        break
                                    else:
                                        flag = False

                            dupis = np.where(np.array(combi).flatten() == m)[0]
                            for dupi in dupis:
                                q, mod = divmod(dupi, 2)
                                if n == side[q][mod]:
                                    if cost < cost_list[q]:
                                        flag = True
                                        del combi[q]
                                        del side[q]
                                        del cost_list[q]
                                        break
                                    else:
                                        flag = False

                            if flag == True:
                                cost_list.append(cost)                                                                      #cost_listにcostを格納
                                combi.append([i, m])                                                                        #combiに接続する領域番号を格納
                                side.append([j, n])                                                                         #sideに接続する端点番号を格納
        
        u, indiices = np.unique(combi, axis = 0, return_index = True)
        range_list = list(range(0, len(combi)))
        not_common_num = list(set(indiices)^set(range_list))
        for i in not_common_num:
            del combi[i]
            del side[i]

        index_form = [i for i, x in enumerate(first_check_result) if x == 1]

        if len(combi) == 0:
            return combi, side, index_form, [], center_points_list
        #########################################################################################################          

        ###########################################接続する領域をリストに格納####################################
        #posはcombiの何番目と何番目がくっつくかを格納する
        #combiの何番目と何番目が接続されるか、更にその番号の接続を入れ子状に格納していく
        pos, unique_pos = [], []
        side_pos = []
        count = 0
        while 1:
            check = False                                                                       #同じ数字が２つあり、組み合わせとして得られたときTrueとなる
            pos.append([])                                                                      #posに接続する組み合わせを格納する
            side_pos.append([])
            if count == 0:                                                                      #はじめはiposにcombiを代入
                ipos = combi
            else:
                ipos = pos[count-1]                                                             #一つ前のposをiposとして取り出す
            flat_pos = list(chain(*ipos))                                                       #iposを１次元リストに変換

            for i in range(0, line_num):
                icount = flat_pos.count(i)                                                      #0~region_numまでfalt_posの中に何回登場するかチェック(0~2のどれかとなる)
                if icount == 2:                                                                 #icountが2だった場合はposに加える(3つの領域が接続される場合の橋渡しになる領域番号はicountが2である)
                    pos[count].append([y for y, row in enumerate(ipos) if i in row])            #iposの中身を順番に取り出していき、該当の領域番号が含まれるindexをyとして格納している
                    check = True                                                                #posに新たに格納された場合はcheckをTrueとする
            if not check :                                                                      #checkがFalseならば更新無しとして、終了
                del pos[-1]                                                                     #posの最後は空白の要素があるため削除
                break
            count += 1                                                                          #countを進める
            if count == 1:
                continue
            if len(pos[count-1]) == len(pos[count-2]):
                break 
        
        if not pos == []:
            #後に入れたものから順に接続関係を整列させていき、最終的にcombiの中身に対する接続情報を取得する
            for i in range(count-1, 0, -1):                                                             #countから逆順に進めていく
                sum_pos, flat_pos = [], []
                for ipos in pos[i]:                                                                     #posの中身をiposに入れていく
                    sum_pos.append(list(set([pos[i-1][p][t] for p in ipos for t in (0,1)])))            #iposの組み合わせ番号から一つ前のposの組み合わせ番号を取り出す
                    flat_pos.extend([k for k in ipos])                                                  #iposを1次元リストに変換
                [sum_pos.append(pos[i-1][j]) for j, _ in enumerate(pos[i-1]) if not j in flat_pos]      #一つ前のposに含まれていて、iposに含まれていない組み合わせ番号をsum_posに含める
                pos[i-1] = sum_pos                                                                      #sum_posを一つ前のposとして更新

            for elem in pos[0]:
                if not elem in unique_pos:
                    unique_pos.append(elem)

   #########################################################################################################

        # color_img = V.visualize_region(new_region_list)
        # color_img = np.clip(color_img * 255, a_min = 0, a_max = 255).astype(np.uint8)
        # cv2.imwrite("./result/connected_color_img.png", color_img)
        # raise ValueError

        return combi, side, index_form, unique_pos, center_points_list

    def check_connect(self, combi, side, index_form, pos, line_list, check_result, correct_connection_index, skip_index):
        V = Visualize()
        connection_index = []
        series_regions, series_sides = [], []
        for x in pos:
            series_side, series_region, flat_num, connection = [], [], [], []
            [flat_num.append(combi[num]) for num in x]
            flat_num = list(chain(*flat_num))
            counter = Counter(flat_num)
            region_num = [elem for elem in flat_num if counter[elem] < 2]
            if region_num == []:
                for index in x:
                    combi[index] = []
                continue
            else:
                region_num = region_num[0]
            for _ in range(len(x)):
                index = [elem for elem in x if region_num in combi[elem]][0]
                if region_num == combi[index][0]:
                    series_side.append(side[index])
                else:
                    side[index].reverse()
                    series_side.append(side[index])
                next_region_num = [elem for elem in combi[index] if not elem == region_num][0]
                series_region.append([region_num, next_region_num])
                connection.extend([region_num, next_region_num])
                region_num = next_region_num
                combi[index] = []
            series_regions.append(series_region)
            series_sides.append(series_side)
            connection_index.append(connection)
            # print("series_region = ", series_region)
            # print("series_side = ", series_side)

        connected_line_list, connected_index_list = [], []
        interpolate_list = []
        for series_region, series_side in zip(series_regions, series_sides):
            connected_line, connected_index = [], []
            interpolate = []
            first_region = series_region[0][0]
            first_side = series_side[0][0]
            first_region_line = line_list[first_region]
            if first_side == 0:
                pre_end_point = [first_region_line[0][0], first_region_line[0][1]]
                first_region_line.reverse()
            else:
                pre_end_point = [first_region_line[-1][0], first_region_line[-1][1]]
            connected_line.extend(first_region_line)
            connected_index.append(first_region)
            for region, region_side in zip(series_region, series_side):
                current_region = region[1]
                current_side = region_side[1]
                region_line = line_list[current_region]
                if current_side == 0:
                    end_point = [region_line[0][0], region_line[0][1]]
                    other_end_point = [region_line[-1][0], region_line[-1][1]]
                else:
                    end_point = [region_line[-1][0], region_line[-1][1]]
                    other_end_point = [region_line[0][0], region_line[0][1]]
                    region_line.reverse()
                connect_line = self.connect_point(pre_end_point, end_point, [])
                connected_line.extend(connect_line)
                connected_line.extend(region_line)
                connected_index.append(current_region)
                interpolate.extend(connect_line)
                pre_end_point = other_end_point
            connected_line = get_unique_list(connected_line)
            connected_line_list.append(connected_line)
            connected_index_list.append(connected_index)
            interpolate_list.append(interpolate)
                   
        for combi_elem, side_elem in zip(combi, side):
            if combi_elem == []:
                continue
            connection_index.append(combi_elem)
            connected_line, connected_index = [], []
            current_region1 = combi_elem[0]
            current_side1 = side_elem[0]
            region_line1 = line_list[current_region1]
            if current_side1 == 0:
                end_point1 = [region_line1[0][0], region_line1[0][1]]
                region_line1.reverse()
            else:
                end_point1 = [region_line1[-1][0], region_line1[-1][1]]
            connected_line.extend(region_line1)
            connected_index.append(current_region1)
            current_region2 = combi_elem[1]
            current_side2 = side_elem[1]
            region_line2 = line_list[current_region2]
            if current_side2 == 0:
                end_point2 = [region_line2[0][0], region_line2[0][1]]
            else:
                end_point2 = [region_line2[-1][0], region_line2[-1][1]]
                region_line2.reverse()
            connect_line = connect_line = self.connect_point(end_point1, end_point2, [])
            connected_line.extend(connect_line)
            connected_line.extend(region_line2)
            connected_index.append(current_region2)
            connected_line = get_unique_list(connected_line)
            connected_line_list.append(connected_line)
            connected_index_list.append(connected_index)
            interpolate_list.append(connect_line)

        # for elem in not_combi:
        #     connected_line_list.append(line_list[elem])
        correct_line, interpolate_list2 = [], []
        for elem in index_form:
            correct_line.append(line_list[elem])
            interpolate_list2.append([])

        connected_line_length = []
        for line in connected_line_list:
            connected_line_length.append(len(line))

        sub_check_result = self.check_connect_accuracy(connected_line_length)
        correct_index = [i for i, x in enumerate(sub_check_result) if x == 1]
        check_index = []
        for index in correct_index:
            correct_line.append(connected_line_list[index])
            interpolate_list2.append(interpolate_list[index])
            check_index.extend(connected_index_list[index])
            correct_connection_index.append(connection_index[index])

        for index in check_index:
            check_result[index] = 1

        correct_line_zlist = []
        for skel in correct_line:
            correct_line_zlist.append([])
            for point in skel:
                z = int(img_copy[point[0]][point[1]])
                correct_line_zlist[-1].append(z)

        new_check_result = []
        count = 0
        for i, elem in enumerate(check_result):
            flag = 0
            while flag == 0:
                if i+count in skip_index:
                    new_check_result.append(0)
                    count += 1
                else:
                    new_check_result.append(elem)
                    flag = 1
        for i in range(len(skip_index)-count):
            new_check_result.append(0)

        correct_connection_index = [list(dict.fromkeys(l)) for l in correct_connection_index]
            
        return new_check_result, correct_line, interpolate_list2, correct_line_zlist, correct_connection_index
        
    def check_connect_accuracy(self, line_length_list):
        check_result = []
        for length in line_length_list:
            if length >= self.full_length - self.error_length and length <= self.full_length + self.error_length:
                check_result.append(1)
            else:
                check_result.append(0)
        
        return check_result

    def sub_check_connect_accuracy(self, line_length_list):
        check_result = []
        for length in line_length_list:
            if length >= self.full_length - self.error_length and length <= self.full_length + self.error_length:
                check_result.append(1)
            else:
                check_result.append(1)
        
        return check_result

    def checkWhereConnected(self, line_list, i, j, m, n):
        MI = MakeImage()
        simg1 = MI.make_colorimage(line_list[i], 255)
        simg1 += MI.make_colorimage(line_list[m], 255)
        simg1 = cv2.circle(simg1, (line_list[i][-j][1], line_list[i][-j][0]), 3, (255, 0, 0), -1)
        simg1 = cv2.circle(simg1, (line_list[m][-n][1], line_list[m][-n][0]), 3, (255, 0, 0), -1)
        V = Visualize()
        V.visualize_1img(simg1)

def show(combi, side, pos, line_list):
    V, MI, CR = Visualize(), MakeImage(), ConnectRegion()
    color = np.zeros((height, width, 3))
    for line in line_list:
        limg = MI.make_image(line, 1)
        color[limg>0] = [255, 255, 255]
    for c, s in zip(combi, side):
        xy1 = line_list[c[0]][-s[0]]
        xy2 = line_list[c[1]][-s[1]]
        line_xy = CR.connect_point(xy1, xy2, [])
        limg = MI.make_image(line_xy, 1)
        color[limg>0] = [0, 255, 0]
        color = cv2.circle(color, [xy1[1], xy1[0]], 3, (255, 0, 0), -1)
        color = cv2.circle(color, [xy2[1], xy2[0]], 3, (255, 0, 0), -1)
    V.visualize_1img(color)

def measure_area(combi, region_list, skip_idndex):
    MI = MakeImage()
    V = Visualize()
    skiped_region = []
    for i in range(len(region_list)):
        if i in skip_idndex:
            continue
        skiped_region.append([])
        skiped_region[-1].extend(region_list[i])

    region = []
    flat_combi = list(set(itertools.chain.from_iterable(combi)))
    if flat_combi == []:
        for r in skiped_region:
            region.extend(r)
    else:
        for index in flat_combi:
            region.extend(skiped_region[index])
    # V.visualize_1region(region)
    print(len(region))

def measure_length(combi, side, line_list):
    MI, CR, D = MakeImage(), ConnectRegion(), Detect()
    V = Visualize()
    line_image = np.zeros((height, width))
    flat_combi = list(set(itertools.chain.from_iterable(combi)))
    if flat_combi == []:
        for line in line_list:
            limg = MI.make_image(line, 1)
            line_image += limg
    else:
        for index in flat_combi:
            limg = MI.make_image(line_list[index], 1)
            line_image += limg
        for c, s in zip(combi, side):
            xy1 = line_list[c[0]][-s[0]]
            xy2 = line_list[c[1]][-s[1]]
            line_xy = CR.connect_point(xy1, xy2, [])
            limg = MI.make_image(line_xy, 1)
            line_image += limg
    _, _, end_point = D.detect_singularity(line_image)
    sp = list(end_point[0])
    ep = list(end_point[1])
    poi = sp
    line = []
    while 1:
        line.append(poi)
        if poi == ep:
            break
        line_image[poi[0]][poi[1]] = 0
        poi = get_neighbor(poi, line_image)[0]

    length = len(line)

    return length, line    

def calc_curvature_2_derivative(line, length):
    # line = line[::-1]

    curvatures = []
    num = 1
    # interval = length // 10
    interval = 5
    maxi = 15
    dxn = line[15][0] - line[5][0]
    dyn = line[15][1] - line[5][1]
    first_theta = math.degrees(math.atan2(dyn, dxn))
    for i in np.arange(maxi, len(line)-maxi, interval):
        dif_theta = 0
        theta_list = []
        for t in range(5, maxi+1, 5):
            dxp = line[i + t][0] - line[i][0]
            dyp = line[i + t][1] - line[i][1]
            thetaOB = math.degrees(math.atan2(dyp, dxp))
            theta = thetaOB - first_theta
            if theta > 180:
                theta = 360 - theta
            elif theta < -180:
                theta = 360 + theta
            theta_list.append(theta)
        
        for i in range(0, len(theta_list)-1):
            dt = theta_list[i+1] - theta_list[i]
            if abs(dt) > 180:
                theta_list[i+1] *= -1
        dif_theta = np.sum(theta_list)
        dif_theta /= (maxi//5)

        if dif_theta > 180:
            dif_theta = 360 - dif_theta
        elif dif_theta < -180:
            dif_theta = 360 + dif_theta

        if not curvatures == [] and abs(curvatures[-1] - dif_theta) > 180:
            dif_theta *= -1
        curvatures.append(dif_theta)
        num += 1

    x = list(range(num-1))
    print(x)
    print(curvatures)
    plt.plot(x, curvatures)
    plt.show()
    sum_curvatures = np.sum(curvatures)

    return curvatures, sum_curvatures

def calc_curvature_2_derivative2(line, length):
    curvatures = []
    num = 1
    interval = length // 10
    interval = 5
    maxi = 10

    for i in np.arange(maxi, len(line)-maxi, interval):
        dif_theta = 0
        theta_list = []
        for t in range(5, maxi+1, 5):
            dxp = line[i + t][0] - line[i][0]
            dyp = line[i + t][1] - line[i][1]
            dxn = line[i][0] - line[i-t][0]
            dyn = line[i][1] - line[i-t][1]
            thetaOA = math.degrees(math.atan2(dyn, dxn))
            thetaOB = math.degrees(math.atan2(dyp, dxp))
            theta = thetaOB - thetaOA
            if theta > 180:
                theta = 360 - theta
            elif theta < -180:
                theta = 360 + theta
            theta_list.append(theta)
        
        for i in range(0, len(theta_list)-1):
            dt = theta_list[i+1] - theta_list[i]
            if abs(dt) > 180:
                theta_list[i+1] *= -1
        dif_theta = np.sum(theta_list)
        dif_theta /= (maxi//5)

        if dif_theta > 180:
            dif_theta = 360 - dif_theta
        elif dif_theta < -180:
            dif_theta = 360 + dif_theta

        if not curvatures == [] and abs(curvatures[-1] - dif_theta) > 180:
            dif_theta *= -1
        curvatures.append(dif_theta)
        num += 1

    x = list(range(num-1))
    print(x)
    print(curvatures)
    plt.plot(x, curvatures)
    plt.show()
    sum_curvatures = np.sum(curvatures)

    return curvatures, sum_curvatures

def cal_dif_curventure(curventure1, curventure2):
    length2 = len(curventure2)
    dif_len = len(curventure1) - length2
    dif_value_list = []
    for i in range(dif_len+1):
        dif_value = 0
        for j, value2 in enumerate(curventure2):
            dif_value += abs(curventure1[i+j]-value2)
        dif_value_list.append(dif_value/length2)

    print(np.min(dif_value_list))
    print(np.argmin(dif_value_list))

def main(input_image, pcd_matrix_index_image, pcd_matrix, MaxDepth):
    MI = MakeImage()
    V = Visualize()

    global height, width, img_copy, max_depth
    height, width = input_image.shape
    img_copy = input_image.copy()
    max_depth = MaxDepth

    ##################################セグメンテーション############################################
    RG = RegionGrowing(input_image)
    region_list2 = RG.search_seed()
    all_img = np.zeros((height, width))
    for region in region_list2:
        rimg = MI.make_image(region, 1)
        all_img += rimg
    V.visualize_1img(all_img)
    # V.visualize_region(region_list2)
    ################################################################################################

    ##################################領域接続######################################################
    CP = CenterPoint()
    center_points_list = CP.search_center_points(region_list2)
    CR = ConnectRegion()
    rregion, new_center_points_list, skip_index, line_list, line_length_list = CR.skip_by_length(region_list2, center_points_list)
    check_result1, correct_connection_index = CR.first_check(line_list, line_length_list)
    combi, side, index_form, pos, center_points_list = CR.connect_region(new_center_points_list, rregion, check_result1, line_list)
    length, line = measure_length(combi, side, line_list)
    measure_area(combi, region_list2, skip_index)
    curventures, sum_curvatures = calc_curvature_2_derivative(line, length)
    print(curventures)
    print(sum_curvatures)
    print(length)
    V.visualize_1region(line)
    ################################################################################################

if __name__ == "__main__":
    #画像の読み込み
    import prePly2depth
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply_filepath))
    pc = np.asanyarray(pcd.points)
    input_image, matrix_image, pcd_matrix, MaxDepth = prePly2depth.main(pc)
    main(input_image, matrix_image, pcd_matrix, MaxDepth)