from distutils.command import check
from distutils.log import error
from heapq import merge
from multiprocessing.sharedctypes import Value
from pickletools import uint8
from pydoc import visiblename
from re import template
from wsgiref.util import request_uri
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random
from pyrsistent import v
from scipy.fftpack import dst
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
from statistics import mode
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

ply_filepath = data_folder_path / "ply" / "2022-11-23" / "2.ply"
ply_filepath = data_folder_path / "ply" / "SI" / "U-SI" / "1.ply"

#コンフィグファイルの読み込み
# config_path = data_folder_path / "cfg" /"config_file_wireharness.yaml"
config_path = data_folder_path / "cfg" /"config_file_flexible.yaml"
config_path = data_folder_path / "cfg" /"config_file_Ushape.yaml"
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
    seq = [list(x) for x in seq]
    return [x for x in seq if not x in seen and not seen.append(x)]

def min_max_x(x):
    x = np.array(x)
    if len(x) > 0:
        max_x = x.max(keepdims=True)
        min_x = x.min(keepdims=True)
        min_max_x = (x - min_x) / (max_x - min_x)
        return min_max_x
    else:
        return 0

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

    def visualize_branch_point(self, region_skel, branch_point, ad_branch_point, end_point = []):
        ske = region_skel.copy()
        ske = gray2color(ske)
        for xy in branch_point:
            ske[xy[0]][xy[1]] = (1, 0, 0)
        for xys in ad_branch_point:
            for xy in xys:
                ske[xy[0]][xy[1]] = (0,0,1)
        if not end_point == []:
            for xy in end_point:
                ske[xy[0]][xy[1]] = (0, 1, 0)

        self.visualize_1img(ske)
        # return ske

    def visualize_arrow(self,line_list, name):
        color = np.zeros((height, width, 3))
        MI = MakeImage()
        CL = ConnectLine()
        # V = Visualize()

        for line in line_list:
            length = len(line)
            point11 = np.array(line[0])
            point12 = np.array(line[length//10])
            point13 = np.array(line[length//5])
            v1 = ((point11 - point12) + (point12 - point13))//2
            theta1 = int(math.degrees(math.atan2(v1[1], v1[0])))
            theta1 = CL.direction16(theta1)
            point21 = np.array(line[-1])
            point22 = np.array(line[-length//10])
            point23 = np.array(line[-length//5])
            v2 = ((point21 - point22) + (point22 - point23))//2
            theta2 = int(math.degrees(math.atan2(v2[1], v2[0])))
            theta2 = CL.direction16(theta2)
            limg = MI.make_image(line, 1)
            color[limg>0] = [255, 255, 255]
            color = cv2.arrowedLine(color, [point21[1], point21[0]], [point21[1] + int(math.sin(math.radians(theta2))*30), point21[0] + int(math.cos(math.radians(theta2))*30)], (0, 255, 0), 2)
            color = cv2.arrowedLine(color, [point11[1], point11[0]], [point11[1] + int(math.sin(math.radians(theta1))*30), point11[0] + int(math.cos(math.radians(theta1))*30)], (255, 0, 0), 2)

        # V.visualize_1img(color)
        SI = SaveImage()
        SI.save_image(color, name)

class SaveImage():

    def save_image(self, image, image_name):
        cv2.imwrite(str(save_folder_path / image_name) + ".png", image)

    def save_region(self, region_list, img_name):
        color_img = np.zeros((height, width, 3))
        for region in region_list:
            blue = random.random()*255 #青色を0〜1の中でランダムに設定
            green = random.random()*255 #緑色を0〜1の中でランダムに設定
            red = random.random()*255 #赤色を0〜1の中でランダムに設定
            for xy in region:
                color_img[xy[0]][xy[1]] = [blue, green, red] #各領域ごとに異なる色を指定

        if not img_name == []:
            cv2.imwrite(str(save_folder_path / img_name) + ".png", color_img)

        return color_img

    def save_centerpoints(self, center_points_list):
        depth_centerpoint = img_copy.copy()
        depth_centerpoint = gray2color(depth_centerpoint)
        points = self.save_region(center_points_list, [])
        depth_centerpoint = depth_centerpoint + points
        cv2.imwrite(str(save_folder_path / "depth_centerpoint.png"), depth_centerpoint)

    def save_grasp_position(self, LC_skel_list, obj_index, optimal_grasp):
        depth = img_copy.copy()
        depth = gray2color(depth)
        skel = LC_skel_list[obj_index]
        for i in range(0, len(skel), 15):
            depth = cv2.circle(depth, (int(skel[i][1]), int(skel[i][0])), 4, (0, 255, 0), -1)
        grasp = optimal_grasp[0]
        grasp_point = grasp[3]
        theta = -(grasp[1] + 90)
        alpha = 30
        open_width = grasp[4]
        alpha -= open_width*5
        grasp_line_vector = np.array([int(alpha*math.cos(math.radians(theta))), int(-alpha*math.sin(math.radians(theta)))])
        circle_point1 = grasp_point + grasp_line_vector
        circle_point2 = grasp_point - grasp_line_vector
        cv2.line(depth, (int(circle_point1[1]), int(circle_point1[0])), (int(circle_point2[1]), int(circle_point2[0])), (0, 0, 255), 2)
        cv2.circle(depth, (int(circle_point1[1]), int(circle_point1[0])), 4, (0, 0, 255), -1)
        cv2.circle(depth, (int(circle_point2[1]), int(circle_point2[0])), 4, (0, 0, 255), -1)
        cv2.circle(depth, (int(grasp_point[1]), int(grasp_point[0])), 4, (0, 255, 0), -1)
        depth[depth < 1] *= 255
        cv2.imwrite(str(data_folder_path / "result" / "grasp_position.png"), depth)
        cv2.imwrite(str(save_folder_path / "grasp_position.png"), depth)

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
        self.full_area = cfg["full_area"]

    def search_seed(self):
        MI = MakeImage()

        if not self.img.ndim == 2:
            raise ValueError("入力画像は2次元(グレースケール)にしてください")

        region_list = []
        value, seed = self.serch_nonzero(0, 0)
        labeled_img = np.zeros((height, width))
        count = 1
        while value:
            self.img[seed[0]][seed[1]] = 0
            region = self.region_growing([seed], [])
            region.insert(0, seed)
            value, seed = self.serch_nonzero(seed[0], seed[1])
            area = len(region)
            # if area > self.lat:
            #     region_list.append(region)
            if area > self.lat and area <= self.full_area + 200:
                region_list.append(region)
                labeled_img += MI.make_image(region, count)
                count += 1
            if area > self.full_area + 200:
                regions = self.split_region(region)
                region_list.extend(regions)
                for region in regions:
                    labeled_img += MI.make_image(region, count)
                    count += 1
        
        
        return region_list, labeled_img

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

    def split_region(self, region):
        MI = MakeImage()
        ###########画像の微分################
        rimg = MI.make_image(region, 1)
        iimg = self.img_copy * rimg
        kernel_x = np.array([[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]])
        kernel_y = np.array([[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]])
        gray_x = cv2.filter2D(iimg, cv2.CV_64F, kernel_x)
        gray_y = cv2.filter2D(iimg, cv2.CV_64F, kernel_y)
        dst = np.sqrt(gray_x**2 + gray_y**2)
        ######################################

        dst[dst < 5] = 0
        dst[dst >= 5] = 1
        dst_skel = skeletonize(dst)
        ###########邪魔な線を消し、領域の境界線をつなぐ###############
        while 1:
            dst_skel, end = self.shave_endpoint(dst_skel)
            if end == []:
                break
        dst_skel[dst_skel > 0] = 1
        ###############################################################

        dst_skel = cv2.dilate(dst_skel.astype(np.uint8), np.ones((3, 3)))
        rimg[dst_skel > 0] = 0
        rimg = self.extract_region(rimg, 30)

        return rimg
        
    def shave_endpoint(self, dst_skel):
        ########端点と孤立点を求める(孤立点は削除)################
        end, lone = self.detect_end_lone_point(dst_skel)
        for xy in lone:
            dst_skel[xy[0]][xy[1]] = 0
        ##########################################################

        ########端点を削っていく(今の状態だと細かい枝が多い)######
        print("451:枝を削る順番に注意、枝先の枝からやるべき")
        for xy in end:
            line = []
            while 1:
                dst_skel[xy[0]][xy[1]] = 0
                line.append(xy)
                neighbor = get_neighbor(xy, dst_skel)
                #分岐点で削除をやめる
                if not len(neighbor) == 1:
                    if len(neighbor) > 1:
                        for nxy in neighbor:
                            #繋がっていた線を切ってしまった場合、復元する
                            if np.sum(dst_skel[nxy[0]-1:nxy[0]+2, nxy[1]-1:nxy[1]+2]) == 2:
                                dst_skel[xy[0]][xy[1]] = 1
                                break
                    break
                xy = neighbor[0]

            #削除した枝がある長さ以上なら復元し、枝を伸ばす
            if len(line) > 10:
                for lxy in line:
                    dst_skel[lxy[0]][lxy[1]] = 1
                p1 = np.array(line[0])
                p2 = np.array(line[9])
                dst_skel = self.extend_line(dst_skel, p1, p2)
                p1 = np.array(line[-1])
                if np.sum(dst_skel[p1[0]-1:p1[0]+2, p1[1]-1:p1[1]+2]) == 2:
                    p2 = np.array(line[-10])
                    dst_skel = self.extend_line(dst_skel, p1, p2)
        dil_dst_skel = self.dilate(dst_skel)
        dst_skel = skeletonize(dil_dst_skel, method = "lee")

        return dst_skel, end
    
    def extend_line(self, dst_skel, p1, p2):
        vector = p1 - p2
        norm_vector = vector / np.linalg.norm(vector, ord=2) 
        alpha = 1
        while 1:
            nextp = [round(p1[0] + alpha * norm_vector[0]), round(p1[1] + alpha * norm_vector[1])]
            if nextp[0] == 0 or nextp[0] == height or nextp[1] == 0 or nextp[1] == width:
                return dst_skel
            if dst_skel[nextp[0]][nextp[1]] == 1:
                break
            alpha += 1
        dst_skel = cv2.line(dst_skel.astype(np.uint8), [p1[1], p1[0]], [nextp[1], nextp[0]], 1, 1)
        
        return dst_skel

    def detect_end_lone_point(self, dst_skel):
        end_point, lonely_point = [], [] #1点連結分岐点、終点、3点連結分岐点を格納する配列
        dst_skel[dst_skel>0] = 1 #値を1に統一
        nozeros = list(zip(*np.where(dst_skel > 0))) #値が1の座標を探索
        #値が1となる座標の近傍9点の値を足し合わせる
        for xy in nozeros:
            point = np.sum(dst_skel[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]) #近傍3×3画素の範囲で画素値の合計を求める
            if point == 2:
                end_point.append(xy) #end_pointに終点座標xyを格納
            if point == 1:
                lonely_point.append(xy)

        return end_point, lonely_point
    
    def dilate(self, dst_skel):
        dst_skel[dst_skel>1] = 1
        nozero = np.nonzero(dst_skel)
        x_min = np.min(nozero[0])
        x_max = np.max(nozero[0])
        y_min = np.min(nozero[1])
        y_max = np.max(nozero[1])
        new_dst_skel = np.zeros((height, width))
        for i in range(x_min-1, x_max+2):
            for j in range(y_min-1, y_max+2):
                kernel = dst_skel[i-1:i+2, j-1:j+2]
                if np.sum(kernel) > 2:
                    new_dst_skel[i][j] = 1
        return new_dst_skel
 
    def delete_small_region(self, image, threshold):
        # _, input_image2 = cv2.threshold(self.img.astype(np.uint8), 15, 255, cv2.THRESH_BINARY)
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(image)
        sizes = stats[1:, -1]
        for i in range(1, nlabels):
            if threshold > sizes[i - 1]:
                image[labels == i] = 0
        return image

    def extract_region(self, rimg, threshold):
        nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(rimg)
        sizes = stats[1:, -1]
        regions = []
        for i in range(1, nlabels):
            if threshold < sizes[i - 1]:
                image = np.zeros((height, width))
                image[labels == i] = 1
                image = cv2.dilate(image, np.ones((3, 3)))
                region = np.argwhere(image > 0)
                region = [list(xy) for xy in region]
                regions.append(np.argwhere(labels == i))

                # region = np.argwhere(labels == i)
                # region = [list(xy) for xy in region]
                # regions.append(np.argwhere(labels == i))
                
        return regions

class Skeletonize():
    
    def __init__(self):
        self.ct = cfg["cut_threshold"]
        self.full_length = cfg["full_length"] 
        # self.line_length_threshold = self.full_length*0.06
        self.line_length_threshold = 20

    def skeletonize_region_list(self, region_list):
        MI, V, D = MakeImage(), Visualize(), Detect()

        kernel_size = 3
        scount = 0
        skel_list2, branch_point_list2, ad_branch_point_list2, end_point_list2 = [], [], [], []
        skip_index, skel2region_index, last_region_index = [], [], []
        for i, region in enumerate(region_list):
            # if not i == 3:
            #     continue
            reg_img = MI.make_image(region, 1)
            reg_img = cv2.morphologyEx(reg_img, cv2.MORPH_CLOSE, np.ones((kernel_size, kernel_size), np.uint8))
            region_skel = skeletonize(reg_img, method="lee")
            region_skel, line_list, skip_flag = self.line_circle_delete(region_skel)
            if not line_list == []:
                skel = []
                for line in line_list:
                    limg = MI.make_image(line, 1)
                    ad_branch_point, branch_point, branch_end_point = D.detect_singularity(limg)
                    # V.visualize_branch_point(limg, branch_point, ad_branch_point, branch_end_point)
                    ske, flag = self.cut_branch(limg, ad_branch_point, branch_point, branch_end_point)
                    skel.extend(ske)
                flag = 1
            else:
                ad_branch_point, branch_point, branch_end_point = D.detect_singularity(region_skel)
                # V.visualize_branch_point(region_skel, branch_point, ad_branch_point, branch_end_point)
                skel, flag = self.cut_branch(region_skel, ad_branch_point, branch_point, branch_end_point)
                if skip_flag == 1:
                    flag = 1
            if flag == 0:
                skel = skel[0]
                if len(skel) <= self.line_length_threshold:
                    skip_index.append(i)
                    continue
                simg = MI.make_image(skel, 1)
                branch_point, ad_branch_point, end_point = D.detect_singularity(simg)
                skel_list2.append(skel)
                branch_point_list2.append(branch_point)
                ad_branch_point_list2.append(ad_branch_point)
                end_point_list2.append(end_point)
                skel2region_index.append(i)
                scount += 1
            else:
                skip_index.append(i)
                last_region_index.append(i)
                for ske in skel:
                    if len(ske) <= self.line_length_threshold:
                        continue
                    simg = MI.make_image(ske, 1)
                    branch_point, ad_branch_point, end_point = D.detect_singularity(simg)
                    skel_list2.append(ske)
                    branch_point_list2.append(branch_point)
                    ad_branch_point_list2.append(ad_branch_point)
                    end_point_list2.append(end_point)
                    skel2region_index.append(i)
                    scount += 1

        # V.visualize_region(skel_list2)
        # raise ValueError       

        return skel_list2, branch_point_list2, ad_branch_point_list2, end_point_list2, skip_index, skel2region_index, last_region_index
    
    def line_circle_delete(self, line_img):
        D = Detect()
        contours, _ = cv2.findContours(line_img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) #輪郭を検出(輪郭の内部に輪郭があれば、その細線には周回する部分がある)

        #周回部分がなければそのまま返す
        if len(contours) == 1:
            return line_img, [], 0

        #周回部分削除(contours[0]は最外部の輪郭)
        large_area = []
        check = 0
        for i in range(1, len(contours)):
            if len(contours[i]) > 100:
                check += 1
                continue
            line_img = cv2.drawContours(line_img, contours, i, 1, -1) #周回部分の内側を塗りつぶす
        line_img = skeletonize(line_img, method="lee") #再度細線化

        # V = Visualize()
        # V.visualize_1img(line_img)

        if check > 0:
            ad_branch_point, branch_point, _ = D.detect_singularity(line_img)
            bps = [ad_points[0] for ad_points in ad_branch_point]
            bps.extend(branch_point)

            for bp in bps:
                line_img_copy = line_img.copy()
                line_img_copy = cv2.circle(line_img_copy, (bp[1], bp[0]), 5, 0, -1)
                contours, _ = cv2.findContours(line_img_copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                clen = [len(c) for c in contours]
                contours = list(contours)
                del contours[np.argmax(clen)]
                del clen[np.argmax(clen)]
                flag = 0
                for i in range(1, len(contours)):
                    if clen[i] > 100:
                        # clist = [c[0] for c in contours[i]]
                        flag += 1
                        break
                if flag == 0:
                    break

            line_list = D.detect_line(line_img_copy)

            # V.visualize_1img(line_img_copy)

            if len(line_list) > 1:
                return [], line_list, 1
            else:
                return line_img_copy, [], 1

        return line_img, [], 0

    def cut_branch(self, skeleton, ad_branch_point, branch_point, branch_end_point):
        ad_bra = list(points[0] for points in ad_branch_point) #ad_branch_pointから代表として1点取得
        index, ad_index = 0, 0 #branch_pointとad_branch_point用の添字
        delete_point = []
        D, MI = Detect(), MakeImage()

        #branch_pointとad_braに値がある間ループ
        while len(branch_point) > 0 or len(ad_bra) > 0:

            #branch_pointに値がある場合
            if len(branch_point) > 0:
                xy = branch_point[index] #注目する1点連結の分岐点を取得
                skeleton, branch_point, ad_bra, ad_branch_point, _, delete_point = self.cutting(skeleton, xy, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point) #cuttingで1点連結の分岐点の場合の枝切り

            #ad_braに値がある場合
            if len(ad_bra) > 0:
                xy = ad_bra[ad_index] #注目する3点連結の分岐点を取得
                skeleton, branch_point, ad_bra, ad_branch_point, _, delete_point = self.ad_cutting(skeleton, xy, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point) #ad_cuttingで3点連結の分岐点の場合の枝切り

        for point in delete_point:
            skeleton = cv2.circle(skeleton, [point[1], point[0]], 5, 0, -1)

        ad_branch_point, branch_point, _ = D.detect_singularity(skeleton)
        #分岐点が存在する場合、分岐点を処理する
        if len(branch_point) > 0: 
            for count in range(0, len(branch_point)//2):
                point, point2 = branch_point[count], branch_point[count+1]
                for y in range(point[0]-1, point[0]+2):
                    for x in range(point[1]-1, point[1]+2):
                        if (y, x) == point:
                            continue
                        if abs(y-point2[0]) + abs(x-point2[1]) == 1:
                            skeleton[y][x] = 0
                            if point in branch_point:
                                branch_point.remove(point)
                                branch_point.remove(point2)
        
        skel = D.detect_line(skeleton)

        if len(skel) > 1:
            return skel, 1
        else:
            return skel, 0

    def cutting(self, skeleton, xy, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point):        
        delete_length = 0

        if not xy in branch_point:
            return skeleton, branch_point, ad_bra, ad_branch_point, 0, delete_point
        
        #branch_pointに入っている座標xyがすでにskeletonから消えている場合は、branch_pointから削除して返却
        if skeleton[xy[0]][xy[1]] == 0:
            branch_point.remove(xy) #branch_pointから座標xyを削除
            return skeleton, branch_point, ad_bra, ad_branch_point, 0, delete_point

        line_img_copy = np.uint8(skeleton.copy()) #skeletonをコピーして、line_img_copyとする
        line_img_copy[xy[0]][xy[1]] = 0 #分岐点を削除

        nlabels, line_labels, stats, _ = cv2.connectedComponentsWithStats(line_img_copy) #ラベリングする

        #ラベルが3つより多ければその分岐点にはまだ枝が残っている
        while len(stats) > 3:
            skeleton, stats, branch_point, ad_bra, ad_branch_point, limg_list, delete_point = self.cut_branch_on_branch(skeleton, line_labels, nlabels, stats, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point)
            line_labels[skeleton == 0] = 0
            min_label1 = self.search_min_line(stats, branch_end_point, limg_list)
            if min_label1 >= 0:
                if stats[min_label1][4] < self.ct:
                    limg = limg_list[min_label1-1]
                    skeleton[limg > 0] = 0
                    delete_length = stats[min_label1][4]
                    nlabels -= 1
                    stats = np.delete(stats, min_label1, 0) #statsからmin_label1番目の行を削除
                else:
                    delete_point.append(xy)
                    delete_length = 0
                    stats = []

                if xy in branch_point:
                    branch_point.remove(xy)

            else:
                return skeleton, branch_point, ad_bra, ad_branch_point, delete_length, delete_point
        
        #whileの条件であるstatsに3つより多くラベルが存在するを満たさない分岐点の処理(分岐点を削除しても線が2本しかできない場合)
        if xy in branch_point:
            nearby = 0 #1点分岐点の近くに1点分岐点があるかのフラグ

            #近傍点探索
            for i in range(-1, 2):
                for j in range(-1, 2):
                    #斜め方向に1点分岐点がある場合
                    if i != 0 and j != 0 and (xy[0]+i, xy[1]+j) in branch_point:
                        #隣接する1点分岐点と注目している1点分岐点の両方に隣接する点を削除
                        skeleton[xy[0]][xy[1]+j] = 0
                        skeleton[xy[0]+i][xy[1]] = 0
                        branch_point.remove((xy[0]+i, xy[1]+j)) #隣接する1点分岐点をbranch_pointから削除
                        nearby += 1

                    #上下左右に1点分岐点がある場合
                    if (i == 0 or j == 0) and (i, j) != (0, 0) and (xy[0]+i, xy[1]+j) in branch_point:
                        line_img_copy[xy[0]+i][xy[1]+j] = 0 #隣接する1点分岐点を削除

                        #通常の枝切り手法と同じ
                        nlabels, line_labels, stats, _ = cv2.connectedComponentsWithStats(line_img_copy)
                        while len(stats) > 3:
                            skeleton, stats, branch_point, ad_bra, ad_branch_point, limg_list, delete_point = self.cut_branch_on_branch(skeleton, line_labels, nlabels, stats, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point)

                            line_labels[skeleton == 0] = 0 
                            min_label1 = self.search_min_line(stats, branch_end_point, limg_list)
                            if min_label1 >= 0:
                                if stats[min_label1][4] < self.ct:
                                    limg = limg_list[min_label1-1]
                                    skeleton[limg > 0] = 0
                                    delete_length = stats[min_label1][4]
                                    nlabels -= 1
                                    stats = np.delete(stats, min_label1, 0) #statsからmin_label1番目の行を削除
                                else:
                                    delete_point.append(xy)
                                    delete_length = 0

                                if xy in branch_point:
                                    branch_point.remove(xy)
                            else:
                                return skeleton, branch_point, ad_bra, ad_branch_point, delete_length, delete_point
                           
                        #2点の1点分岐点のどちらを削除するか
                        point1 = np.sum(skeleton[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]) #注目している1点分岐点の近傍点数を計算
                        point2 = np.sum(skeleton[xy[0]+i-1:xy[0]+i+2, xy[1]+j-1:xy[1]+j+2]) #隣接している1点分岐点の近傍点数を計算
                        #注目している1点分岐点の近傍点が3点なら消す
                        if point1 == 3 and point2 > 3:
                            skeleton[xy[0]][xy[1]] = 0
                        #隣接している1点分岐点の近傍点が3点なら消す
                        elif point2 == 3 and point1 > 3:
                            skeleton[xy[0]+i][xy[1]+j] = 0

            #1点分岐点が隣接していない場合
            if nearby == 0:
                # raise ValueError
                around_branch = skeleton[xy[0]-1:xy[0]+2, xy[1]-1:xy[1]+2]
                if np.sum(around_branch) > 3:
                    #通常の枝切り手法と同じ
                    nlabels, line_labels, stats, _ = cv2.connectedComponentsWithStats(line_img_copy)
                    while len(stats) > 3:
                        skeleton, stats, branch_point, ad_bra, ad_branch_point, limg_list, delete_point = self.cut_branch_on_branch(skeleton, line_labels, nlabels, stats, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point)
                        line_labels[skeleton == 0] = 0 
                        min_label1 = self.search_min_line(stats, branch_end_point, limg_list)
                        if min_label1 >= 0:
                            if stats[min_label1][4] < self.ct:
                                # limg = np.zeros((height, width))
                                # limg[line_labels == min_label1] = 1
                                limg = limg_list[min_label1-1]
                                skeleton[limg > 0] = 0
                                delete_length = stats[min_label1][4]
                                # line_labels[line_labels > min_label1] -= 1
                                nlabels -= 1
                                stats = np.delete(stats, min_label1, 0) #statsからmin_label1番目の行を削除
                            else:
                                delete_point.append(xy)
                                delete_length = 0

                            if xy in branch_point:
                                branch_point.remove(xy)

        if xy in branch_point:
            branch_point.remove(xy)
        
        return skeleton, branch_point, ad_bra, ad_branch_point, delete_length, delete_point

    def cut_branch_on_branch(self, skeleton, line_labels, nlabels, stats, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point):
        MI = MakeImage()
        limg_list = []
        #各線ごとに分岐点が含まれるか確かめ、含まれている場合はその分岐線から削除する
        for i in range(1, nlabels+1):
            label_line = list(zip(*np.where(line_labels == i)))
            
            brabranch = list(set(branch_point) & set(label_line)) #min_labelの枝に1点連結の枝が含まれていれば、その座標をbrabranchに格納
            ad_brabranch = list(set(ad_bra) & set(label_line)) #min_labelの枝に3点連結の枝が含まれていれば、その座標をad_brabranchに格納
            limg = MI.make_image(label_line, 1)

            #brabranchに値が格納されている場合
            if len(brabranch) > 0:
                #brabranchに入っている座標の数だけ回す
                for xy2 in brabranch:
                    if self.check_branch_point(limg, xy2):
                        continue
                    skeleton -= limg
                    limg, branch_point, ad_bra, ad_branch_point, delete_length, delete_point = self.cutting(limg, xy2, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point) #cuttingにbrabranchの座標を分岐点として枝切り
                    stats[i][4] -= delete_length
                    skeleton += limg

            #ad_brabranchに値が格納されている場合
            if len(ad_brabranch) > 0:
                #ad_brabranchに入っている座標の数だけ回す
                for xy2 in ad_brabranch:
                    if self.check_branch_point(limg, xy2):
                        continue
                    skeleton -= limg
                    limg, branch_point, ad_bra, ad_branch_point, delete_length, delete_point = self.ad_cutting(limg, xy2, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point) #ad_cuttingにad_brabranchの座標を分岐点として枝切り
                    stats[i][4] -= delete_length
                    skeleton += limg

            limg_list.append(limg)

        return skeleton, stats, branch_point, ad_bra, ad_branch_point, limg_list, delete_point

    def check_branch_point(self, limg, branch_point):
        kernel = limg[branch_point[0]-1:branch_point[0]+2, branch_point[1]-1:branch_point[1]+2]
        if np.sum(kernel) < 4:
            return True
        else:
            return False

    def search_min_line(self, stats, branch_end_point, limg_list):
        sort_index_list = np.argsort(stats[:, 4])
        sort_index_list = np.delete(sort_index_list, -1)
        for index in sort_index_list:
            limg = limg_list[index-1]
            cood_list = list(zip(*np.where(limg > 0)))
            dup = list(set(branch_end_point) & set(cood_list)) 
            if not dup == []:
                return index

        return -1

    def select_cut_branch(self, nlabels, limg_list, branch_point, branch_end_point):
        line_list, end_point, len_list, end_point2 = [], [], [], []

        limg_list_copy = limg_list.copy()
        no_ep_index = []
        count = 0
        for i in range(0, nlabels-1):
            limg = limg_list_copy[i].copy()
            # V = Visualize()
            # V.visualize_1img(limg)
            if self.branch_end_point_check(limg, branch_end_point):
                no_ep_index.append(count)
            for point in branch_point:
                # xy = get_neighbor(point, limg)[0]
                # break
                try:
                    xy = get_neighbor(point, limg)[0]
                    break
                except IndexError:
                    continue
            
            end_point.append(xy)
            line = self.extract_line(xy, limg)
            end_point2.append(line[-1])
            len_list.append(len(line))
            line_list.append(line)
            count += 1

        return line_list, end_point2
        
    def branch_end_point_check(self, limg, branch_end_point):
        for point in branch_end_point:
            if limg[point[0]][point[1]] > 0:
                return False
        return True

    def extract_line(self, xy, limg):
        line = []
        while 1:
            line.append(xy)
            limg[xy[0]][xy[1]] = 0
            neighbor = get_neighbor(xy, limg)
            try:
                xy = neighbor[0]
            except IndexError:
                break

        return line

    def calculate_direction(self, line, start_index, length):
        if length  >= start_index + 10:
            p1, p2 = line[start_index], line[start_index + 10]
        elif length < start_index + 10 and length > 15:
            start_index = 5
            p1, p2 = line[start_index], line[start_index + 10]
        elif length <= 15:
            p1, p2 = line[0], line[-1]
        vector = [p1[0]-p2[0], p1[1]-p2[1]]
        theta = int(math.degrees(math.atan2(vector[1], vector[0])))

        return theta

    def direction16(self, theta):
        for i in range(0, 16):
            angle = -180 + 22.5*i
            if angle < theta and theta <= angle + 22.5:
                theta = angle + 11.25

        return theta

    def ad_cutting(self, skeleton, xy, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point):
        delete_length = 0

        if not xy in ad_bra:
            return skeleton, branch_point, ad_bra, ad_branch_point, 0, delete_point
        coods = ad_branch_point[ad_bra.index(xy)] #座標xyが含まれている3点分岐点をad_branch_pointから取得する

        #ad_bra, ad_branch_pointに入っている座標xyがすでにskeletonから消えている場合は、ad_bra, aad_branch_pointから削除して返却
        if skeleton[xy[0]][xy[1]] == 0:
            ad_bra.remove(xy) #ad_braから座標xyを削除
            ad_branch_point.remove(coods) #ad_branch_pointから座標群coodsを削除
            return skeleton, branch_point, ad_bra, ad_branch_point, 0, delete_point

        for cood in coods:
            kernel = skeleton[cood[0]-1:cood[0]+2, cood[1]-1:cood[1]+2]
            if np.sum(kernel) <= 3:
                return skeleton, branch_point, ad_bra, ad_branch_point, 0, delete_point

        x, y = [], []
        line_img_copy = np.uint8(skeleton.copy()) #skeletonをコピーして、line_img_copyとする
        #coodsに入っている座標をx座標とy座標に分け、各座標を削除
        for cood in coods:
            line_img_copy[cood[0]][cood[1]] = 0 #line_img_copyからcoodsの座標を削除

        nlabels, line_labels, stats, _ = cv2.connectedComponentsWithStats(line_img_copy) #ラベリング

        #ラベルが3つより多ければその分岐点にはまだ枝が残っている
        while len(stats) > 3:
            skeleton, stats, branch_point, ad_bra, ad_branch_point, limg_list, delete_point = self.cut_branch_on_branch(skeleton, line_labels, nlabels, stats, branch_point, ad_bra, ad_branch_point, branch_end_point, delete_point)
            line_labels[skeleton == 0] = 0 
            min_label1 = self.search_min_line(stats, branch_end_point, limg_list)
            if min_label1 >= 0:
                if stats[min_label1][4] < self.ct:
                    limg = limg_list[min_label1 - 1]
                    skeleton[limg > 0] = 0
                    delete_length = stats[min_label1][4]
                    nlabels -= 1
                    stats = np.delete(stats, min_label1, 0) #statsからmin_label1番目の行を削除

                    value = []
                    for cood in coods:
                        value.append(np.sum(skeleton[cood[0]-1:cood[0]+2, cood[1]-1:cood[1]+2]))
                    check_value = np.where(np.array(value) >= 4)[0] #値が4以上となるvalueの添字を取得
                    ad_value = np.where(np.array(value) == 3)[0] #値が3となるvalueの添字を取得
                    if len(check_value) == 2 and len(ad_value) > 0:
                        skeleton[coods[ad_value[0]][0]][coods[ad_value[0]][1]] = 0 #値が3となる分岐点を削除
                else: 
                    delete_point.append(xy)
                    stats = []

                if xy in ad_bra:  
                    ad_bra.remove(xy)
                    ad_branch_point.remove(coods)
            else:
                return skeleton, branch_point, ad_bra, ad_branch_point, delete_length, delete_point

        return skeleton, branch_point, ad_bra, ad_branch_point, delete_length, delete_point

class Sort():

    def sort_skel_list(self, skel_list, branch_point_list, ad_branch_point_list, end_point_list):
        new_skel_list = [[] for i in range(0, len(skel_list))]
        for i, skel in enumerate(skel_list):
            ######################今回使うデータの整理#####################################
            try:
                flat_ad_branch = sum(ad_branch_point_list[i], [])
                flat_ad_branch = [list(e) for e in flat_ad_branch]
            except IndexError:
                flat_ad_branch = []
            try:
                branch_point = [list(e) for e in branch_point_list[i]]
            except IndexError:
                branch_point = []
            end_point = [list(e) for e in end_point_list[i]]
            ###############################################################################

            #########端点が分岐点に含まれているかを判断後、最初の注目点を決定##############
            if len(end_point) > 2:
                for ep in end_point:
                    _, _, flag = self.check_neighbor(ep, branch_point)
                    if flag == 0:
                        poi = ep
                        break
            else:
                poi = list(end_point[0])
            ###############################################################################
            # print("branch_point = ",branch_point)
            # print("flat_ad_branch = ",flat_ad_branch)
            #########################近傍点探索、リストに順番に格納########################
            skel = [list(elem) for elem in skel]
            skel.remove(poi)
            new_skel_list[i].append(poi)
            while skel != []:
                #3点分岐がある場合
                if poi in flat_ad_branch:
                    flag = 0
                    poi, skel, flat_ad_branch, new_skel_list[i] = self.ad_sort(poi, skel, new_skel_list[i], flat_ad_branch)

                #1点分岐がある場合
                if poi in branch_point:
                    branch_point.remove(poi)
                    poi, skel, new_skel_list[i], branch_point = self.branch_sort(poi, skel, new_skel_list[i], branch_point, end_point_list[i])
  
                for xy in skel:
                    check_x = abs(poi[0] - xy[0])
                    check_y = abs(poi[1] - xy[1])
                    if check_x <= 1 and check_y <= 1 and check_x + check_y <= 2:
                        new_skel_list[i].append(xy)
                        poi = xy
                        skel.remove(xy)
                        break

            # if len(end_point_list[i]) == 1:
            #     end_point_list[i].append(new_skel_list[i][-1])

        # new_skel_list = self.cut_line(new_skel_list)

        return new_skel_list

    #近傍点がpoint_listに含まれているかの判断(注目点、point_listに含まれている近傍点、フラグが返される)
    def check_neighbor(self, poi, point_list):
        for i in range(-1, 2):
            for j in range(-1, 2):
                if [poi[0]+i, poi[1]+j] in point_list:
                    return poi, [poi[0]+i, poi[1]+j], 1
        
        return poi, [0, 0], 0
    
    def branch_sort(self, poi, skel, new_skel_list, branch_point, end_point_list):
        for t in range(-1, 2):
            for k in range(-1, 2):
                next_poi = [poi[0]+t, poi[1]+k]
                if next_poi in end_point_list:
                    skel.remove(next_poi)
                    continue
                if not next_poi in branch_point:
                    continue
                new_skel_list.append(next_poi)
                skel.remove(next_poi)
                branch_point.remove(next_poi)
                for l in range(-1, 2):
                    for m in range(-1, 2):
                        nearby_poi = [poi[0]+l, poi[1]+m]
                        if nearby_poi == poi or nearby_poi == next_poi:
                            continue
                        check_x = np.abs(next_poi[0] - nearby_poi[0])
                        check_y = np.abs(next_poi[1] - nearby_poi[1])
                        if check_x <= 1 and check_y <= 1 and check_x + check_y <= 2 and nearby_poi in skel:
                            skel.remove(nearby_poi)
                poi = next_poi
                break
            else:
                continue
            break
        
        return poi, skel, new_skel_list, branch_point

    def ad_sort(self, poi, skel, new_skel_list, flat_ad_branch):
        flat_ad_copy = flat_ad_branch.copy()

        while 1:
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    neighbor = [poi[0]+i, poi[1]+j]
                    if not neighbor in flat_ad_copy and neighbor in skel:
                        next_poi = neighbor
                        new_skel_list.append(poi)
                        skel.remove(next_poi)
                        flat_ad_branch, skel = self.delete_ad_branch(poi, flat_ad_branch, skel)
                        return next_poi, skel, flat_ad_branch, new_skel_list
            
            flat_ad_copy.remove(poi)
            if poi in skel:
                skel.remove(poi)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i == 0 and j == 0:
                        continue
                    neighbor = [poi[0]+i, poi[1]+j]
                    if neighbor in flat_ad_copy:
                        poi = neighbor
                        new_skel_list.append(poi)
                        if poi in skel:
                            skel.remove(poi)
                        break
                else:
                    continue
                break

    def delete_ad_branch(self, poi, flat_ad_branch, skel):
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor = [poi[0]+i, poi[1]+j]
                if neighbor in flat_ad_branch:
                    flat_ad_branch.remove(neighbor)
                    if neighbor in skel:
                        skel.remove(neighbor)
                    flat_ad_branch, skel = self.delete_ad_branch(neighbor, flat_ad_branch, skel)
        return flat_ad_branch, skel

    def cut_line(self, line_list):
        cut_line_list = []
        for line in line_list:
            length = len(line)
            del line[-length//15:0]
            del line[0:length//15]
            cut_line_list.append(line)
        return cut_line_list

    def first_check(self, sorted_skel_list, skel2region_index):
        full_length = cfg["full_length"]
        error_length  = full_length//5
        first_skip_index, first_skip_skel = [], []
        new_sorted_skel_list = []
        new_skel2region_index = skel2region_index.copy()
        for i, skel in enumerate(sorted_skel_list):
            length = len(skel)
            if length>full_length-error_length and length<full_length+error_length:
                first_skip_index.append(i)
                first_skip_skel.append(skel)
            else:
                new_sorted_skel_list.append(skel)
        
        first_skip_skel2region = []
        for i in reversed(first_skip_index):
            first_skip_skel2region.append(new_skel2region_index[i])
            del new_skel2region_index[i]
                
        first_skip_skel2region = first_skip_skel2region[::-1]

        return new_sorted_skel_list, new_skel2region_index, first_skip_skel, first_skip_skel2region

def sunder_curve(skel_list, skel2region_index):
    sunder_idnex, sunder_info = [], []

    for j, skel in enumerate(skel_list):
        length = len(skel)
        curvatures, curvatures_index = [], []
        interval = 5
        curve_threshold = 50

        V = Visualize()
        D = Detect()
        MI = MakeImage()
        skel_color = MI.make_image(skel, 255)
        skel_color_copy = skel_color.copy()

        for i in np.arange(interval, length-interval, interval):
            dxn = skel[i][0] - skel[i - interval][0]
            dxp = skel[i + interval][0] - skel[i][0]
            dyn = skel[i][1] - skel[i - interval][1]
            dyp = skel[i + interval][1] - skel[i][1]
            thetaOA = math.degrees(math.atan2(dyn, dxn))
            thetaOB = math.degrees(math.atan2(dyp, dxp))
            dif_theta = thetaOB - thetaOA
            if dif_theta > 180:
                dif_theta = 360 - dif_theta
            elif dif_theta < -180:
                dif_theta = 360 + dif_theta
            curvatures.append(int(dif_theta))
            curvatures_index.append(i)
            # skel_color[skel[i][0]][skel[i][1]] = [255, 0, 0]

        if length - (i+interval) >= int(interval*3/4):
            i += interval
            dxn = skel[i][0] - skel[i - interval][0]
            dxp = skel[-1][0] - skel[i][0]
            dyn = skel[i][1] - skel[i - interval][1]
            dyp = skel[-1][1] - skel[i][1]
            thetaOA = math.degrees(math.atan2(dyn, dxn))
            thetaOB = math.degrees(math.atan2(dyp, dxp))
            dif_theta = thetaOB - thetaOA
            if dif_theta > 180:
                dif_theta = 360 - dif_theta
            elif dif_theta < -180:
                dif_theta = 360 + dif_theta
            curvatures.append(int(dif_theta))
            curvatures_index.append(i)
            # skel_color[skel[i][0]][skel[i][1]] = [255, 0, 0]

        # print("max_curvatures = ", np.max(np.abs(curvatures)))
        # print("curvature = ", curvatures)
        abs_curvatures = np.abs(curvatures)
        sorted_curvature_index = np.argsort(abs_curvatures)[::-1]
        max_curvature = np.max(abs_curvatures)

        if max_curvature > curve_threshold:
            sunder_idnex.append(j)
            for curve_index in sorted_curvature_index:
                curve = curvatures[curve_index]
                abs_curve = abs(curve)
                if abs_curve > curve_threshold:
                    overThre_index = curvatures.index(curve)
                    sindex = curvatures_index[overThre_index]
                    point = skel[sindex]
                    skel_color = cv2.circle(skel_color, (point[1], point[0]), 3, [0, 0, 0], -1)
                else:
                    break
            sunder_lines = D.detect_line(skel_color)
            region_index = skel2region_index[j]
            sunder_info.append([j, region_index, sunder_lines])
            # V.visualize_1img(skel_color_copy)
            # V.visualize_region(sunder_lines)

    sunder_info = sunder_info[::-1]
    for info in sunder_info:
        skel_index = info[0]
        region_index = info[1]
        sunder_lines = info[2]
        
        del skel_list[skel_index]
        del skel2region_index[skel_index]
        line_num = 0
        for line in sunder_lines:
            if len(line) > 20:
                skel_list.append(line)
                line_num += 1
        for _ in range(line_num):
            skel2region_index.append(region_index)

    return skel_list, skel2region_index

class ConnectLine():

    def __init__(self):
        self.full_length = cfg["full_length"]
        self.error_length = self.full_length//20
        self.interval = self.full_length//10

    def __main__(self, sorted_skel_list, end_point_list):
        count_list, switch_list, end_point_list, interpolate_list = self.choose_connect_line(sorted_skel_list)
        if not count_list == []:
            CL_skel_list, z_list, end_point_list, interpolate, connection_index_list = self.connect_line(sorted_skel_list, count_list, interpolate_list)
        else:
            CL_skel_list, z_list, end_point_list, interpolate, connection_index_list = [], [], [], [], []

        return CL_skel_list, z_list, interpolate, connection_index_list, end_point_list

    def choose_connect_line(self, skel_list):
        thetas, ends = self.line_orientation(skel_list)

        count_list, switch_list, flat_combi, flat_count = [], [], [], []
        cost, index = [], []
        detail_cost = []
        cost1d = []
        interpolate_list = []
        min_distance_threshold = 10
        cost_threshold = 2
        alpha = 4
        beta = 0.2
        gamma = 4
        delta = 4
        sum_theta_threshold = 1000
        dif_depth_threshold = 1000
        distance_threshold = 1000

        skel_num = len(skel_list)
        full_length = cfg["full_length"]
        #終点同士の距離を求める
        for count in range(0, skel_num-1): #countは細線の番号を表す(end_point_listの第1添字)
            cost.append([[], []])
            index.append([[], []])
            detail_cost.append([[], []])
            for switch in (0, 1): #switchは終点の番号を表す(終点は細線一本につき2点ある)(end_point_listの第2添字)
                end1 = ends[count][switch][0:2]
                theta1 = thetas[count][switch]
                depth1 = ends[count][switch][2]
                for i in range(count+1, skel_num): #ペアとなる点が含まれる細線番号(end_point_listの第1添字)
                    for j in range(0, 2): #ペアとなる点の終点の番号(end_point_listの第2添字)
                        end2 = ends[i][j][0:2]
                        theta2 = thetas[i][j]
                        depth2 = ends[i][j][2]

                        sum_theta = np.abs(np.abs(theta1 - theta2) - 180)
                        if sum_theta > 90:
                            continue
                        sum_theta /= 180

                        dif_depth = np.abs(depth1 - depth2)
                        dif_depth /= 255

                        distance = int(np.sqrt((end1[0]-end2[0])**2 + (end1[1]-end2[1])**2))
                        if distance > 100:
                            continue
                        cost_distance = distance / full_length

                        vx12 = end2[0] - end1[0]
                        vy12 = end2[1] - end1[1]
                        theta12 = int(math.degrees(math.atan2(vy12, vx12)))
                        dif_theta1 = np.abs(theta12 - theta1)
                        if dif_theta1 > 180:
                            dif_theta1 = 360 - dif_theta1
                        theta21 = int(math.degrees(math.atan2(-vy12, -vx12)))
                        dif_theta2 = np.abs(theta21 - theta2)
                        if dif_theta2 > 180:
                            dif_theta2 = 360 - dif_theta2
                        dif_theta = (dif_theta1 + dif_theta2)/360     

                        if distance <= min_distance_threshold:
                            par_cost = alpha*cost_distance + beta*sum_theta + gamma*dif_depth//3 + delta*dif_depth
                        else:
                            if sum_theta < 0.3 and dif_theta > 0.5:
                                dif_theta += 20
                            par_cost = alpha*cost_distance + beta*sum_theta + gamma*dif_theta + delta*dif_depth

                        cost[count][switch].append(par_cost)
                        index[count][switch].append([i, j]) #countが注目細線のインデックス、switchが端点のインデックス、iとjがペアとなる細線のインデックスと端点のインデックス
                        detail_cost[count][switch].append([alpha*cost_distance, beta*sum_theta, gamma*dif_theta, delta*dif_depth])
                        cost1d.append([count, switch, i, j, par_cost])
                        # if par_cost < 1.5:
                        #     print("start")
                        #     print("count, switch = ", count, switch)
                        #     print("i, j = ", i, j)
                        #     print("par_cost = ", par_cost)
                        #     print("detail_cost = ", [cost_distance, sum_theta, dif_theta, dif_depth])
                        #     print("detail_cost2 = ", [alpha*cost_distance, beta*sum_theta, gamma*dif_theta, delta*dif_depth])
                        #     print("")

                        #     MI = MakeImage()
                        #     limg = MI.make_image(skel_list[count], 1)
                        #     limg += MI.make_image(skel_list[i], 1)
                        #     limg = cv2.line(limg, (end1[1], end1[0]), (end2[1], end2[0]), 1, 1)
                        #     plt.imshow(limg)
                        #     plt.show()
        cost.append([[], []])
        index.append([[], []])

        sorted_cost1d = sorted(cost1d, key = lambda x:x[4])

        correct_count_index_list = []
        for cinfo in sorted_cost1d:
            count = cinfo[0]
            switch = cinfo[1]
            vs_count = cinfo[2]
            vs_switch = cinfo[3]
            par_cost = cinfo[4]
            if any(c in correct_count_index_list for c in (count, vs_count)):
                continue
            if any(c in count_list for c in ([count, vs_count], [vs_count, count])):
                continue
            if any(c in flat_combi for c in ([count, switch], [vs_count, vs_switch])):
                continue
            if par_cost > cost_threshold:
                break
            skel1 = skel_list[count]
            end1 = ends[count][switch][0:2]
            skel2 = skel_list[vs_count]
            end2 = ends[vs_count][vs_switch][0:2]
            line, interpolate = self.connect2skels(skel1, skel2, list(end1), list(end2))
            length = len(line)
            if self.full_length + self.error_length < length:
                continue

            check = 0#過去に接続した線が含まれているかのフラグ
            explored = [count, vs_count]#今回の接続する線のindex
            flat_count_copy = flat_count.copy()#すでに接続した線のindex
            count_list_index_count = []
            count_list_index_vs_count = []

            if count in flat_count:#今回の接続する線が過去に接続候補として上がっているか
                check += 1
                cindex_count = count
                while 1:
                    cindex = flat_count_copy.index(cindex_count)#今回の線のflat_countにおけるindex
                    count_list_index_count.append(cindex//2)#flat_countの半分がcount_listのindexになる
                    flat_count_copy[cindex] = -1
                    #過去にcountの線と接続することになった線のflat_countにおけるindexを求める
                    if cindex % 2 == 0:
                        vs_cindex = cindex + 1
                    else:
                        vs_cindex = cindex - 1
                    cindex_count = flat_count_copy[vs_cindex]#過去の相方の線のindex
                    flat_count_copy[vs_cindex] = -1
                    explored.append(cindex_count)#接続することになった３つの線を加える
                    if not cindex_count in flat_count_copy:
                        break
        
            if vs_count in flat_count_copy:
                check += 1
                cindex_vs_count = vs_count
                while 1:
                    cindex = flat_count_copy.index(cindex_vs_count)
                    count_list_index_vs_count.append(cindex//2)
                    flat_count_copy[cindex] = -1
                    if cindex % 2 == 0:
                        vs_cindex = cindex + 1
                    else:
                        vs_cindex = cindex - 1
                    cindex_vs_count = flat_count_copy[vs_cindex]
                    flat_count_copy[vs_cindex] = -1
                    explored.append(cindex_vs_count)
                    if not cindex_vs_count in flat_count_copy:
                        break

            flat_count.append(count)
            flat_count.append(vs_count)
            flat_combi.append([count, switch])
            flat_combi.append([vs_count, vs_switch])
            count_list.append([count, vs_count])
            switch_list.append([switch, vs_switch])
            interpolate_list.append(interpolate)

            # V = Visualize()
            # MI = MakeImage()
            # limg = np.zeros((height, width, 3))
            # limg += MI.make_colorimage(skel_list[flat_count[-2]], 1)
            # limg += MI.make_colorimage(skel_list[flat_count[-1]], 1)
            # point1 = skel_list[flat_count[-2]][-1*switch_list[-1][0]]
            # point2 = skel_list[flat_count[-1]][-1*switch_list[-1][1]]
            # limg = cv2.circle(limg, (point1[1], point1[0]), 3, [255,0,0],-1)
            # limg = cv2.circle(limg, (point2[1], point2[0]), 3, [255, 0, 0], -1)
            # print(flat_count[-2], flat_count[-1])
            # print(switch_list[-1][0], switch_list[-1][1])
            # V.visualize_1img(limg)

            #３つの線を繋いで長さを求める
            if check > 0:
                count_list_index = count_list_index_count[::-1]#逆順に変更
                current_count_list_index = len(count_list)-1#現在のcount_listのindexを取得
                count_list_index.append(current_count_list_index)
                count_list_index.extend(count_list_index_vs_count)#接続先の線と接続することが決まっている線のcount_listのindex
                end_count = [ei for ei in explored if flat_count.count(ei) == 1]#端に位置する線のindex
                if end_count == []:
                    del flat_count[-1]
                    del flat_count[-1]
                    del flat_combi[-1]
                    del flat_combi[-1]
                    del count_list[-1]
                    del switch_list[-1]
                    del interpolate_list[-1]
                else:
                    current_count = [i for i in count_list[count_list_index[0]] if i in end_count][0]
                    connected_count_list = [current_count]
                    line = skel_list[current_count].copy()
                    interpolate = interpolate_list[count_list_index[0]]
                    line_end, _ = self.which_is_neighbor(line, interpolate)
                    if line_end == 0:
                        line = line[::-1]
                    for ci in count_list_index:
                        interpolate = interpolate_list[ci]
                        current_count = [i for i in count_list[ci] if not i == current_count][0]
                        connected_count_list.append(current_count)
                        current_line = skel_list[current_count]
                        line_end, interpolate_end = self.which_is_neighbor(current_line, interpolate)
                        if interpolate_end == 0:
                            interpolate = interpolate[::-1]
                        line.extend(interpolate)
                        if line_end == -1:
                            current_line = current_line[::-1]
                        line.extend(current_line)
                    length = len(line)
                    if length > self.full_length+self.error_length:
                        del flat_count[-1]
                        del flat_count[-1]
                        del flat_combi[-1]
                        del flat_combi[-1]
                        del count_list[-1]
                        del switch_list[-1]
                        del interpolate_list[-1]
                    elif length <= self.full_length+self.error_length and  length >= self.full_length-self.error_length:
                        correct_count_index_list.extend(connected_count_list)

        return count_list, switch_list, ends, interpolate_list

    def cal_curvature_of_connected_several_lines(self, skel_list, explored, count_list, count_list_index, interpolate_list, end_count, current_count_list_index):
        current_count = [i for i in count_list[count_list_index[0]] if i in end_count][0]
        connected_count_list = [current_count]
        line = skel_list[current_count].copy()
        interpolate = interpolate_list[count_list_index[0]]
        line_end, _ = self.which_is_neighbor(line, interpolate)
        if line_end == 0:
            line = line[::-1]
        line_separate = [line.copy()]
        for ci in count_list_index:
            interpolate = interpolate_list[ci]
            current_count = [i for i in count_list[ci] if not i == current_count][0]
            connected_count_list.append(current_count)
            current_line = skel_list[current_count]
            line_end, interpolate_end = self.which_is_neighbor(current_line, interpolate)
            if interpolate_end == 0:
                interpolate = interpolate[::-1]
            line.extend(interpolate)
            line_separate.append(interpolate)
            if line_end == -1:
                current_line = current_line[::-1]
            line.extend(current_line)
            line_separate.append(current_line)
        length = len(line)

        if length > self.full_length + self.error_length:
            return [], current_count_list_index
            
        curvature = self.calc_curvature(line)
        flag, value = self.cal_dif_curvature(curvature, 1)

        if flag == 1:
            if self.full_length - self.error_length < length and length < self.full_length + self.error_length:
                return connected_count_list, []
            return [], []
        else:
            line_separate_copy = line_separate.copy()
            del line_separate_copy[0]
            del line_separate_copy[0]
            line0 = [point for separate in line_separate_copy for point in separate]
            curvature0 = self.calc_curvature(line0)
            flag, value0 = self.cal_dif_curvature(curvature0)

            line_separate_copy = line_separate.copy()
            del line_separate_copy[-1]
            del line_separate_copy[-1]
            line1 = [point for separate in line_separate_copy for point in separate]
            curvature1 = self.calc_curvature(line1)
            flag, value1 = self.cal_dif_curvature(curvature1)

            if value0 < value1:
                return [], count_list_index[-1]
            else:
                return [], count_list_index[0]

    def which_is_neighbor(self, line, interpolate):
        lends = [line[0], line[-1]]
        iends = [interpolate[0], interpolate[-1]]

        for i, lend in enumerate(lends):
            for j, iend in enumerate(iends):
                if abs(lend[0]-iend[0])<=1 and abs(lend[1]-iend[1])<=1:
                    return -i, -j 
        raise ValueError

    def cal_dif_curvature(self, curvature2, flag=0):
        curvature1 = self.curvature
        length2 = len(curvature2)
        dif_len = len(curvature1) - length2
        dif_value_list_minus, dif_value_list_plus, max_dif_value = [], [], []
        
        if dif_len < 0:
            curvature2_plus = curvature2
            curvature2_minus = list(np.array(curvature2) * -1)
            for sign in (1, -1):
                if sign == 1:
                    curvature2 = curvature2_plus
                elif sign == -1:
                    curvature2 = curvature2_minus
                for i in range(-1*dif_len+1):
                    dif_value = 0
                    for j, value2 in enumerate(curvature1):
                        value = abs(curvature2[i+j]-value2)
                        dif_value += value
                    if sign == 1:
                        dif_value_list_plus.append(dif_value/length2)
                    elif sign == -1:
                        dif_value_list_minus.append(dif_value/length2)
                    max_dif_value.append(sign)
        else:
            curvature1_plus = curvature1
            curvature1_minus = list(np.array(curvature1) * -1)
            for sign in (1, -1):
                if sign == 1:
                    curvature1 = curvature1_plus
                elif sign == -1:
                    curvature1 = curvature1_minus
                for i in range(dif_len+1):
                    dif_value = 0
                    for j, value2 in enumerate(curvature2):
                        value = abs(curvature1[i+j]-value2)
                        dif_value += value
                    if sign == 1:
                        dif_value_list_plus.append(dif_value/length2)
                    elif sign == -1:
                        dif_value_list_minus.append(dif_value/length2)
                    max_dif_value.append(sign)
        
        min_index_plus = np.argmin(dif_value_list_plus)
        min_index_minus = np.argmin(dif_value_list_minus)
        if dif_value_list_plus[min_index_plus] < dif_value_list_minus[min_index_minus]:
            min_value = dif_value_list_plus[min_index_plus]
            # min_index = min_index_plus
            # curvature1 = curvature1_plus
        else:
            min_value = dif_value_list_minus[min_index_minus]
            # min_index = min_index_minus
            # curvature1 = curvature1_minus

        # if flag == 1:
        #     print("min_value = ", min_value)
        #     x = list(range(len(curvature1)))
        #     x2 = list(range(min_index,min_index+length2))
        #     if np.max(x) < np.max(x2):
        #         raise ValueError
        #     fig, ax = plt.subplots()
        #     ax.plot(x, curvature1, color="red")
        #     ax.plot(x2, curvature2, color="blue")
        #     plt.show()

        if min_value < 12:
            # print("min_value = ", min_value)
            # x = list(range(len(curvature1)))
            # x2 = list(range(min_index,min_index+length2))
            # if np.max(x) < np.max(x2):
            #     raise ValueError
            # fig, ax = plt.subplots()
            # ax.plot(x, curvature1, color="red")
            # ax.plot(x2, curvature2, color="blue")
            # plt.show()

            return 1, min_value
        return 0, min_value

    def calc_curvature(self, line):
        curvatures = []
        num = 1
        length = len(line)
        interval = 5
        maxi = 10
        dxn = line[15][0] - line[5][0]
        dyn = line[15][1] - line[5][1]
        first_theta = math.degrees(math.atan2(dyn, dxn))
        for i in np.arange(maxi, length-maxi, interval):
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

        # x = list(range(num-1))
        # print(x)
        # print(curvatures)
        # plt.plot(x, curvatures)
        # plt.show()

        return curvatures

    def connect2skels(self, skel1, skel2, end1, end2):
        CR = ConnectRegion()

        connect_line_list = []
        if skel1[0] == end1:
            skel1 = skel1[::-1]
        elif not skel1[-1] == end1:
            raise ValueError("1461 端点が違います") 
        connect_line_list.extend(skel1)
        connect_line = CR.connect_point(end1, end2, [])
        if connect_line[0] == end1 or connect_line[0] == end2:
            del connect_line[0]
        if connect_line[-1] == end1 or connect_line[-1] == end2:
            del connect_line[-1]
        connect_line_list.extend(connect_line)
        if skel2[-1] == end2:
            skel2 = skel2[::-1]
        elif not skel2[0] == end2:
            raise ValueError("1470 端点が違います") 
        connect_line_list.extend(skel2)

        return connect_line_list, connect_line

    def line_orientation(self, skel_list):
        thetas, ends = [], []
        depth_img = img_copy.copy()

        for i, skel in enumerate(skel_list):
            thetas.append([])
            ends.append([])
            length = len(skel)
            
            point11 = np.array(skel[0])
            point12 = np.array(skel[length//10])
            point13 = np.array(skel[length//5])
            v1 = ((point11 - point12) + (point12 - point13))//2
            theta1 = int(math.degrees(math.atan2(v1[1], v1[0])))
            theta1 = self.direction16(theta1)
            roi = depth_img[point11[0]-1:point11[0]+2, point11[1]-1:point11[1]+2]
            if len(roi) == 0 or list(roi[0]) == []:
                modified_point = self.modify_point(point11)
                roi = depth_img[modified_point[0]-1:modified_point[0]+2, modified_point[1]-1:modified_point[1]+2]
                point11 = np.array(skel[0])
            depth1 = np.max(roi)
            if depth1 == 0:
                print("check point!!")
            thetas[-1].append(theta1)
            ends[-1].append([point11[0], point11[1], depth1])

            point21 = np.array(skel[-1])
            point22 = np.array(skel[-length//10])
            point23 = np.array(skel[-length//5])
            v2 = ((point21 - point22) + (point22 - point23))//2
            theta2 = int(math.degrees(math.atan2(v2[1], v2[0])))
            theta2 = self.direction16(theta2)
            roi2 = depth_img[point21[0]-1:point21[0]+2, point21[1]-1:point21[1]+2]
            if len(roi2) == 0 or list(roi2[0]) == []:
                modified_point = self.modify_point(point21)
                roi2 = depth_img[modified_point[0]-1:modified_point[0]+2, modified_point[1]-1:modified_point[1]+2]
                point21 = np.array(skel[-1])
            depth2 = np.max(roi2)
            if depth2 == 0:
                print("check point!!")
            thetas[-1].append(theta2)
            ends[-1].append([point21[0], point21[1], depth2])  

        V = Visualize()
        V.visualize_arrow(skel_list, "skel_orientation")
        
        return thetas, ends

    def direction16(self, theta):
        for i in range(0, 16):
            angle = -180 + 22.5*i
            if angle < theta and theta <= angle + 22.5:
                theta = angle + 11.25

        return theta
    
    def modify_point(self, point):
        if point[0] == 0:
            point[0] += 1
        if point[1] == 0:
            point[1] += 1
        if point[0] == height-1:
            point[0] -= 1
        if point[1] == width-1:
            point[1] -= 1
        
        return point

    def connect_line(self, skel_list, count_list, interpolate_list):
        depth = img_copy.copy()
        depth = cv2.morphologyEx(depth, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        connected_skel_list, connected_end_point_list, connected_z_list, connected_interpolate_list, connection_index = [], [], [], [], []
        separate_list = []
        flat_count = list(itertools.chain.from_iterable(count_list))
        end_line = [x for x in flat_count if flat_count.count(x) == 1]
        for ei in end_line:
            if ei == -1:
                continue
            connected_skel, connected_interpolate, connection = [], [], []
            separate = []
            while 1:
                if not ei in flat_count:
                    break
                flat_index = flat_count.index(ei)
                cindex = flat_index//2
                line = skel_list[ei]
                interpolate = interpolate_list[cindex]
                line_edge, interpolate_edge = self.which_is_neighbor(line, interpolate)
                if line_edge == 0:
                    line = line[::-1]
                if interpolate_edge == -1:
                    interpolate = interpolate[::-1]
                connected_skel.extend(line)
                connected_skel.extend(interpolate)
                separate.append(line)
                separate.append(interpolate)
                connected_interpolate.extend(interpolate)
                connection.append(count_list[cindex])
                ei = [x for x in count_list[cindex] if not x == ei][0]
                if ei in end_line:
                    end_line[end_line.index(ei)] = -1
                flat_count[cindex*2] = -1
                flat_count[cindex*2+1] = -1
            line = skel_list[ei]
            if abs(line[-1][0]-connected_skel[-1][0])<=1 and abs(line[-1][1]-connected_skel[-1][1])<=1:
                line = line[::-1]
            connected_skel.extend(line)
            separate.append(line)
            connected_skel_list.append(connected_skel)
            connection_index.append(connection)
            connected_interpolate_list.append(connected_interpolate)
            connect_ep = []
            for i in range(0, len(separate)-2, 2):
                connect_ep.append([separate[i][-1], separate[i+2][0]])
            connected_end_point_list.append(connect_ep)

        for line in connected_skel_list:
            z = []
            for point in line:
                z.append(depth[point[0]][point[1]])
            connected_z_list.append(z)
        
        return connected_skel_list, connected_z_list, connected_end_point_list, connected_interpolate_list, connection_index  

    def region_img_index1(self, index_list1, index_list2, skel_skip_index, skip_index):
        skel_skip_index.sort()
        skip_index.sort()
        index_list2 = self.insert2list(index_list2, skel_skip_index)
        index_list1 = self.insert2list(index_list1, skip_index)
        flat_index1 = self.make_1and2DimentionalList2flat(index_list1)
        flat_index1.sort()
        index_list2 = self.insert2list(index_list2, flat_index1)
        index_list2.extend(index_list1)

        return index_list2

    def region_img_index2(self, index_list1, len_skel_list, skel_skip_index, skip_index):
        index_list2 = []
        for i in range(len_skel_list):
            index_list2.append(i)
        skel_skip_index.sort()
        skip_index.sort()
        index_list2 = self.insert2list(index_list2, skel_skip_index)
        index_list1 = self.insert2list(index_list1, skip_index)
        flat_index1 = self.make_1and2DimentionalList2flat(index_list1)
        flat_index1.sort()
        index_list2 = self.insert2list(index_list2, flat_index1)
        index_list2.extend(index_list1)

        return index_list2

    def insert2list(self, target_list, insert_index):
        for x in insert_index:
            for i, li in enumerate(target_list):
                if type(li) is list:
                    for j, y in enumerate(li):
                        if y >= x:
                            target_list[i][j] += 1
                else:
                    if li >= x:
                        target_list[i] += 1
        return target_list

    def make_1and2DimentionalList2flat(self, target_list):
        flat = []
        for li in target_list:
            if type(li) is list:
                for x in li:
                    flat.append(x)
            else:
                flat.append(li)
        return flat

    def connect_check(self, CL_skel_list, CL_index_list, CL_connect_ep):
        correct_skel_index, correct_CL_index, connect_skel_index, connect_ep = [], [], [], []
        for i, skel in enumerate(CL_skel_list):
            length = len(skel)
            if length>=self.full_length-self.error_length and length<=self.full_length+self.error_length:
                correct = CL_index_list[i]
                correct_skel_index.extend(correct)
                correct_CL_index.append(i)
            elif length < self.full_length - self.error_length:
                connect = CL_index_list[i]
                ep = CL_connect_ep[i]
                connect_skel_index.append(connect)
                connect_ep.append(ep)

        return correct_skel_index, correct_CL_index, connect_skel_index, connect_ep

    def which_region_left(self, skip_region_list, skel2region_index, first_skip_skel2region_index, last_region_index, first_skip_skel, correct_skel_index):
        flat = []
        
        # if not first_skip_skel2region_index == []:
        #     raise ValueError("check first_skip_skel2region_index")

        for i in range(len(first_skip_skel)):
            region_index = first_skip_skel2region_index[i]
            if not region_index in skip_region_list:
                skip_region_list.append(region_index)

        flat_skel_index = []
        for indexes in correct_skel_index:
            for index in indexes:
                flat_skel_index.append(index)
                region_index = skel2region_index[index]
                if not region_index in skip_region_list:
                    skip_region_list.append(region_index)

        last_skel_index = []
        for region_index in last_region_index:
            skel_index = [i for i, si in enumerate(skel2region_index) if si == region_index]
            for si in skel_index:
                if not si in flat_skel_index:
                    last_skel_index.append(si)

        return skip_region_list, last_skel_index

class CenterPoint():

    def __init__(self):
        self.Size = 51
        self.Step = 16
        self.half_S = self.Size//2
        self.nextPdist = 6 #Don't change(loop process will go wrong)

    def search_center_points(self, region_list, skip_region_index):
        line_templates = [[] for i in range(0, self.Step)]
        center_points_list, cp2region_index = [], []
        MI = MakeImage()

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
        for i, region in enumerate(region_list):
            if i in skip_region_index:
                continue
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
            cp2region_index.append(i)
        
        return center_points_list, cp2region_index

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
        self.error_length = self.full_length//20
        self.interval = self.full_length//10
        self.line_length_threshold = self.full_length*0.08

    def point2line(self, center_points_list, last_index, sorted_skel_list, cp2region_index, skel2region_index):
        new_center_points_list, line_list, new_cp2region_index = [], [], []
        #中心点の選別と点を結んで線形化
        for i, center_point in enumerate(center_points_list):
            if len(center_point) > 3:
                line = self.connect_points(center_point)
                new_center_points_list.append(center_point)
                line_list.append(line)
                new_cp2region_index.append(cp2region_index[i])
        
        #中心点のindexを入力すると細線のindexが帰ってくる(cp2skel_index)
        cp2skel_index = []
        for cri in new_cp2region_index:
            rsi = skel2region_index.index(cri)
            cp2skel_index.append(rsi)
        
        for i in last_index:
            line_list.append(sorted_skel_list[i])
            cp2skel_index.append(i)

        # line_list = self.cut_line(line_list)
                
        return new_center_points_list, line_list, cp2skel_index

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
            ends[-1][0] = point11
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
            ends[-1][1] = point21
            depthes[-1].append(int(depth))

        V = Visualize()
        V.visualize_arrow(line_list, "region_orientation")
        return thetas, ends, depthes

    def match_index(self, ends, connect_skel_index, connect_ep, cp2skel_index, sorted_skel_list, line_list, thetas, depthes):
        connect_ep_index, connect_cp_index = [], []
        for indexes, eps in zip(connect_skel_index, connect_ep):
            connect_ep_index.append([])
            connect_cp_index.append([])
            for index, ep in zip(indexes, eps):
                connect_ep_index[-1].append([])
                connect_cp_index[-1].append([])
                for i in range(2):
                    connect_index = index[i]
                    try:
                        sc_index = cp2skel_index.index(connect_index)
                    except ValueError:
                        line_list.append(sorted_skel_list[connect_index])
                        cp2skel_index.append(connect_index)
                        sc_index = len(cp2skel_index)-1
                        theta, end, depth = self.end_point_information([sorted_skel_list[connect_index]])
                        ends.append(end[0])
                        thetas.append(theta[0])
                        depthes.append(depth[0])
                    cp_end = ends[sc_index]
                    end_point = ep[i]
                    ep_dist = []
                    for cpe in cp_end:
                        ep_dist.append(math.dist(cpe, end_point))
                    connect_ep_index[-1][-1].append(np.argmin(ep_dist))
                    connect_cp_index[-1][-1].append(sc_index)

        return connect_ep_index, connect_cp_index, line_list, ends, thetas, depthes

    def connect_region(self, center_points_list, line_list, thetas, ends, depthes, connect_skel_index, connect_ep_index):
        line_num = len(line_list)                                                                                   #region_numに領域の数を代入
        combi, side, cost_list = [], [], []                                                        
        
        connect_skel_index_flat = []
        for indexes in connect_skel_index:
            for index in indexes:
                connect_skel_index_flat.append(index)
        
        connect_ep_index_flat = []
        for indexes in connect_ep_index:
            for index in indexes:
                connect_ep_index_flat.append(index)

        # print(connect_skel_index_flat)
        # print(connect_ep_index_flat)
        # show(connect_skel_index_flat, connect_ep_index_flat, line_list)
        # input()

        skel_ep_comb = []
        for si, ei in zip(connect_skel_index_flat, connect_ep_index_flat):
            skel_ep_comb.append([si[0], ei[0]])
            skel_ep_comb.append([si[1], ei[1]])

        ################################用語集######################################################
        #sum_theta : 2つの端点の向きの合計(互いに向き合っているときが最小となる)
        #dif_theta : 2つの端点を結んだ線の向きと端点の向きの差分
        #distance : 2つの端点の距離
        #dif_depth : 2つの端点の深度の差分
        ###########################################################################################

        ##################################接続条件############################################################
        count_list = connect_skel_index_flat
        switch_list = connect_ep_index_flat
        flat_combi = [[count, switch] for counts, switches in zip(connect_skel_index_flat, connect_ep_index_flat) for count, switch in zip(counts, switches)]
        flat_count = list(itertools.chain.from_iterable(connect_skel_index_flat))

        CL = ConnectLine()
        interpolate_list = []
        for i, (count, switch) in enumerate(zip(count_list, switch_list)):  
            line1 = line_list[count[0]]
            line2 = line_list[count[1]]
            end1 = ends[count[0]][switch[0]]
            end2 = ends[count[1]][switch[1]]
            _, interpolate = CL.connect2skels(line1, line2, list(end1), list(end2))
            interpolate_list.append(interpolate)

        cost_list, index_list = [], []
        detail_cost = []
        cost1d = []
        min_distance_threshold = 10                                                                                        #distanceに対する閾値
        cost_threshold = 1.5
        alpha = 4
        beta = 0.2
        gamma = 2
        delta = 4
        CL = ConnectLine()

        #thetas,ends,center_point_depthの情報からsum_theta,dif_theta,distance,dif_depthを求め、コストを計算することで接続の有無を取得する
        for count in range(0, line_num-1): 
            cost_list.append([[], []])
            index_list.append([[], []])
            detail_cost.append([[], []])                                                                                #対象となる1つ目の領域番号
            for switch in (0, 1): 
                if [count, switch] in skel_ep_comb:
                    continue                                                                                               #該当する領域の端点番号(端点は1つの領域に2つ)
                end1 = ends[count][switch]
                theta1 = thetas[count][switch]                                                                                       
                depth1 = depthes[count][switch]
                for m in range(count+1, line_num):                                                                            #対象となる2つ目の領域番号(全領域、端点を総当りで調べる)
                    for n in (0, 1):   
                        if [m, n] in skel_ep_comb:
                            continue     
                        if [count, m] in connect_skel_index_flat or [m, count] in connect_skel_index_flat:
                            continue                                                                                #該当する領域の端点番号(端点は1つの領域に2つ)
                        end2 = ends[m][n]
                        theta2 = thetas[m][n]
                        depth2 = depthes[m][n]
                        
                        sum_theta = np.abs(np.abs(theta1 - theta2) - 180)
                        if sum_theta > 90:
                            continue
                        sum_theta /= 180

                        distance = int(np.sqrt((end1[0]-end2[0])**2 + (end1[1]-end2[1])**2))                                #distanceの計算
                        if distance > 100:
                            continue
                        cost_distance = distance / self.full_length

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

                        if distance <= min_distance_threshold:
                            par_cost = alpha*cost_distance + beta*sum_theta + gamma*dif_theta//3 + delta*dif_depth                                #distanceが閾値より小さければ、dif_depthを3分の1にする
                        else:
                            if sum_theta < 0.3 and dif_theta > 0.5:
                                dif_theta += 20
                            par_cost = alpha*cost_distance + beta*sum_theta + gamma*dif_theta + delta*dif_depth
                        
                        cost_list[count][switch].append(par_cost)
                        index_list[count][switch].append([m, n])
                        detail_cost[count][switch].append([alpha*cost_distance, beta*sum_theta, gamma*dif_theta, delta*dif_depth])
                        cost1d.append([count, switch, m, n, par_cost])

        cost_list.append([[], []])
        index_list.append([[], []])

        sorted_cost1d = sorted(cost1d, key = lambda x:x[4])
        correct_count_index_list = []
        for cinfo in sorted_cost1d:
            count = cinfo[0]
            switch = cinfo[1]
            vs_count = cinfo[2]
            vs_switch = cinfo[3]
            par_cost = cinfo[4]
            if any(c in correct_count_index_list for c in (count, vs_count)):
                continue
            if any(c in count_list for c in ([count, vs_count], [vs_count, count])):
                continue
            if any(c in flat_combi for c in ([count, switch], [vs_count, vs_switch])):
                continue
            if par_cost > cost_threshold:
                break
            skel1 = line_list[count]
            end1 = ends[count][switch][0:2]
            skel2 = line_list[vs_count]
            end2 = ends[vs_count][vs_switch][0:2]
            line, interpolate = CL.connect2skels(skel1, skel2, list(end1), list(end2))
            length = len(line)
            if self.full_length + self.error_length < length:
                continue

            check = 0#過去に接続した線が含まれているかのフラグ
            explored = [count, vs_count]#今回の接続する線のindex
            flat_count_copy = flat_count.copy()#すでに接続した線のindex
            count_list_index_count = []
            count_list_index_vs_count = []

            if count in flat_count:#今回の接続する線が過去に接続候補として上がっているか
                check += 1
                cindex_count = count
                while 1:
                    cindex = flat_count_copy.index(cindex_count)#今回の線のflat_countにおけるindex
                    count_list_index_count.append(cindex//2)#flat_countの半分がcount_listのindexになる
                    flat_count_copy[cindex] = -1
                    #過去にcountの線と接続することになった線のflat_countにおけるindexを求める
                    if cindex % 2 == 0:
                        vs_cindex = cindex + 1
                    else:
                        vs_cindex = cindex - 1
                    cindex_count = flat_count_copy[vs_cindex]#過去の相方の線のindex
                    flat_count_copy[vs_cindex] = -1
                    explored.append(cindex_count)#接続することになった３つの線を加える
                    if not cindex_count in flat_count_copy:
                        break
        
            if vs_count in flat_count_copy:
                check += 1
                cindex_vs_count = vs_count
                while 1:
                    cindex = flat_count_copy.index(cindex_vs_count)
                    count_list_index_vs_count.append(cindex//2)
                    flat_count_copy[cindex] = -1
                    if cindex % 2 == 0:
                        vs_cindex = cindex + 1
                    else:
                        vs_cindex = cindex - 1
                    cindex_vs_count = flat_count_copy[vs_cindex]
                    flat_count_copy[vs_cindex] = -1
                    explored.append(cindex_vs_count)
                    if not cindex_vs_count in flat_count_copy:
                        break

            flat_count.append(count)
            flat_count.append(vs_count)
            flat_combi.append([count, switch])
            flat_combi.append([vs_count, vs_switch])
            count_list.append([count, vs_count])
            switch_list.append([switch, vs_switch])
            interpolate_list.append(interpolate)

            # V = Visualize()
            # MI = MakeImage()
            # limg = np.zeros((height, width, 3))
            # limg += MI.make_colorimage(skel_list[flat_count[-2]], 1)
            # limg += MI.make_colorimage(skel_list[flat_count[-1]], 1)
            # point1 = skel_list[flat_count[-2]][-1*switch_list[-1][0]]
            # point2 = skel_list[flat_count[-1]][-1*switch_list[-1][1]]
            # limg = cv2.circle(limg, (point1[1], point1[0]), 3, [255,0,0],-1)
            # limg = cv2.circle(limg, (point2[1], point2[0]), 3, [255, 0, 0], -1)
            # print(flat_count[-2], flat_count[-1])
            # print(switch_list[-1][0], switch_list[-1][1])
            # V.visualize_1img(limg)

            #３つの線を繋いで長さを求める
            if check > 0:
                count_list_index = count_list_index_count[::-1]#逆順に変更
                current_count_list_index = len(count_list)-1#現在のcount_listのindexを取得
                count_list_index.append(current_count_list_index)
                count_list_index.extend(count_list_index_vs_count)#接続先の線と接続することが決まっている線のcount_listのindex
                end_count = [ei for ei in explored if flat_count.count(ei) == 1]#端に位置する線のindex
                if end_count == []:
                    del flat_count[-1]
                    del flat_count[-1]
                    del flat_combi[-1]
                    del flat_combi[-1]
                    del count_list[-1]
                    del switch_list[-1]
                    del interpolate_list[-1]
                else:
                    current_count = [i for i in count_list[count_list_index[0]] if i in end_count][0]
                    connected_count_list = [current_count]
                    line = line_list[current_count].copy()
                    interpolate = interpolate_list[count_list_index[0]]
                    line_end, _ = CL.which_is_neighbor(line, interpolate)
                    if line_end == 0:
                        line = line[::-1]
                    for ci in count_list_index:
                        interpolate = interpolate_list[ci]
                        current_count = [i for i in count_list[ci] if not i == current_count][0]
                        connected_count_list.append(current_count)
                        current_line = line_list[current_count]
                        line_end, interpolate_end = CL.which_is_neighbor(current_line, interpolate)
                        if interpolate_end == 0:
                            interpolate = interpolate[::-1]
                        line.extend(interpolate)
                        if line_end == -1:
                            current_line = current_line[::-1]
                        line.extend(current_line)
                    length = len(line)
                    if length > self.full_length+self.error_length:
                        del flat_count[-1]
                        del flat_count[-1]
                        del flat_combi[-1]
                        del flat_combi[-1]
                        del count_list[-1]
                        del switch_list[-1]
                        del interpolate_list[-1]
                    elif length <= self.full_length+self.error_length and  length >= self.full_length-self.error_length:
                        correct_count_index_list.extend(connected_count_list)

        _, indiices = np.unique(count_list, axis = 0, return_index = True)
        range_list = list(range(0, len(count_list)))
        not_common_num = list(set(indiices)^set(range_list))
        for i in not_common_num:
            del count_list[i]
            del switch_list[i]

        if len(count_list) == 0:
            return count_list, switch_list, [], center_points_list
        # #########################################################################################################          

        ###########################################接続する領域をリストに格納####################################
        #posはcombiの何番目と何番目がくっつくかを格納する
        #combiの何番目と何番目が接続されるか、更にその番号の接続を入れ子状に格納していく
        pos, unique_pos = [], []
        num = 0
        while 1:
            check = False                                                                       #同じ数字が２つあり、組み合わせとして得られたときTrueとなる
            pos.append([])                                                                      #posに接続する組み合わせを格納する
            if num == 0:                                                                      #はじめはiposにcombiを代入
                ipos = count_list
            else:
                ipos = pos[num-1]                                                             #一つ前のposをiposとして取り出す
            flat_pos = list(chain(*ipos))                                                       #iposを１次元リストに変換

            for i in range(0, line_num):
                inum = flat_pos.count(i)                                                      #0~region_numまでfalt_posの中に何回登場するかチェック(0~2のどれかとなる)
                if inum == 2:                                                                 #icountが2だった場合はposに加える(3つの領域が接続される場合の橋渡しになる領域番号はicountが2である)
                    pos[num].append([y for y, row in enumerate(ipos) if i in row])            #iposの中身を順番に取り出していき、該当の領域番号が含まれるindexをyとして格納している
                    check = True                                                                #posに新たに格納された場合はcheckをTrueとする
            if not check :                                                                      #checkがFalseならば更新無しとして、終了
                del pos[-1]                                                                     #posの最後は空白の要素があるため削除
                break
            num += 1                                                                          #countを進める
            if num == 1:
                continue
            if len(pos[num-1]) == len(pos[num-2]):
                break 
        
        if not pos == []:
            #後に入れたものから順に接続関係を整列させていき、最終的にcombiの中身に対する接続情報を取得する
            for i in range(num-1, 0, -1):                                                             #countから逆順に進めていく
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

        return count_list, switch_list, unique_pos, center_points_list

    def check_connect(self, combi, side, pos, line_list):
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

        z_list = []
        for skel in connected_line_list:
            z_list.append([])
            for point in skel:
                z = int(img_copy[point[0]][point[1]])
                z_list[-1].append(z)
            
        return connected_line_list, interpolate_list, z_list, connected_index_list,
        
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

    def merge_CR_and_CL(self, CR_line_list, CR_interpolate, CR_z_list, correct_CL_index, CL_skel_list, CL_interpolate, CL_z_list, first_skip_skel, connect_skel_index,  connect_region_index, line_list):
        for index in correct_CL_index:
            CR_line_list.append(CL_skel_list[index])
            CR_interpolate.append(CL_interpolate[index])
            CR_z_list.append(CL_z_list[index])

        depth = img_copy.copy()
        first_skip_line_index = []
        for skel in first_skip_skel:
            z_list = []
            for xy in skel:
                z_list.append(depth[xy[0]][xy[1]])
            CR_line_list.append(skel)
            CR_interpolate.append([])
            CR_z_list.append(z_list)
            first_skip_line_index.append(len(CR_line_list)-1)

        flat_combi = list(set(itertools.chain.from_iterable(itertools.chain.from_iterable(connect_skel_index))))
        flat_combi.extend(list(set(itertools.chain.from_iterable(connect_region_index))))
        check1, check2 = [], []
        for i, line in enumerate(line_list):
            if not i in flat_combi:
                check1.append(line)
                CR_line_list.append(line)
                CR_interpolate.append([])
                z = []
                for xy in line:
                    z.append(depth[xy[0]][xy[1]])
                CR_z_list.append(z)
            else:
                check2.append(line)

        return CR_line_list, CR_interpolate, CR_z_list, first_skip_line_index

class GaussLinkingIintegral():

    def __init__(self):
        self.full_length = cfg["full_length"]
        self.error_length = self.full_length//5
        self.len_threshold = self.full_length*2//3

    def calculate_GLI(self, skel_list, z_list, interpolate):
        len_skel = len(skel_list)
        GLI_matrix = np.zeros((len_skel, len_skel))
        objs_GLI = np.zeros((len_skel))
        interval = 10

        len_list = [len(skel) for skel in skel_list]

        if len_skel <= 1:
            return [], [], []

        ###########線iと線jからintervalごとに点を取り出しGLIを計算####################################
        for i in range(0, len_skel-1):
            if len_list[i] < self.len_threshold:
                continue
            skel1 = skel_list[i]
            z1 = z_list[i]
            for j in range(i+1, len_skel):
                if len_list[j] < self.len_threshold:
                    continue
                num = 0
                skel2 = skel_list[j]
                z2 = z_list[j]
                for k in range(0, len_list[i]-interval, interval):
                    r1 = np.array([skel1[k][0], skel1[k][1], z1[k]])
                    next_r1 = np.array([skel1[k+interval][0], skel1[k+interval][1], z1[k+interval]])
                    dr1 = next_r1 - r1
                    for l in range(0, len_list[j]-interval, interval):
                        r2 = np.array([skel2[l][0], skel2[l][1], z2[l]])
                        next_r2 = np.array([skel2[l+interval][0], skel2[l+interval][1], z2[l+interval]])
                        dr2 = next_r2 - r2
                        r12 = r2 - r1
                        norm_r12 = np.linalg.norm(r12)
                        num += np.dot(np.cross(dr1, dr2), r12) / (norm_r12**3)
                GLI_matrix[i][j] = abs(num)
                GLI_matrix[j][i] = abs(num)    

        GLI_matrix, cross_matrix = self.recalculate_GLI(skel_list, interpolate, GLI_matrix, z_list)
        for i in range(0, len_skel): 
            objs_GLI_num = np.sum(GLI_matrix[i]) / ((len_list[i]-interval)//interval+1)
            if objs_GLI_num == 0 or math.isnan(objs_GLI_num):
                objs_GLI[i] = 100
            else:
                objs_GLI[i] = objs_GLI_num

        return GLI_matrix, objs_GLI, cross_matrix

    def recalculate_GLI(self, skel_list, interpolate, GLI, z_list):
        MI, V = MakeImage(), Visualize()
        image_list, dilate_image_list = [], []
        cross_matrix = [[[] for i in range(0, len(skel_list))] for j in range(0, len(skel_list))]
        for skel in skel_list:
            image = MI.make_image(skel, 1)
            image_list.append(image)
            dilate_image = cv2.dilate(image, np.ones((3, 3)), iterations=1)
            dilate_image_list.append(dilate_image)

        for i in range(0, len(skel_list)-1):
            dilate_image1 = dilate_image_list[i]
            image1 = image_list[i]
            for j in range(i+1, len(skel_list)):
                dilate_image2 = dilate_image_list[j]
                image2 = image_list[j]
                sum_image = dilate_image1 + dilate_image2
                sum_image[sum_image<2] = 0
                nlabels, labels, stats, _ = cv2.connectedComponentsWithStats(sum_image)
                for num in range(1, nlabels):
                    if stats[num][4] > 5:
                        label = labels.copy()
                        label[label != num] = 0
                        label[label > 0] = 1
                        sum_label1 = label + image1
                        sum_label2 = label + image2
                        cross_point1 = np.argwhere(sum_label1 == 2)
                        try :
                            cross_point1 = list(cross_point1[0])
                        except IndexError:
                            continue                       
                        cross_point2 = np.argwhere(sum_label2 == 2)
                        try:
                            cross_point2 = list(cross_point2[0])
                        except IndexError:
                            continue
                        # V.visualize_2img(sum_label1, sum_label2)

                        if cross_point1 in interpolate[i] and cross_point2 in interpolate[j]:
                            index1 = skel_list[i].index(cross_point1)
                            index2 = skel_list[j].index(cross_point2)
                            z1 = z_list[i][index1]
                            z2 = z_list[j][index2]
                            if z1 > z2:
                                cross_matrix[i][j].append([cross_point1[0], cross_point1[1], 0])
                                cross_matrix[j][i].append([cross_point2[0], cross_point2[1], 1])
                            else:
                                cross_matrix[j][i].append([cross_point2[0], cross_point2[1], 0])
                                cross_matrix[i][j].append([cross_point1[0], cross_point1[1], 1])
                        elif cross_point1 in interpolate[i]:
                            cross_matrix[j][i].append([cross_point2[0], cross_point2[1], 0])
                            cross_matrix[i][j].append([cross_point1[0], cross_point1[1], 1])
                        elif cross_point2 in interpolate[j]:
                            cross_matrix[i][j].append([cross_point1[0], cross_point1[1], 0])
                            cross_matrix[j][i].append([cross_point2[0], cross_point2[1], 1])
                        else:
                            continue
                            # raise ValueError("二物体の位置関係が不明瞭です。recalculate_GLI")

        for i in range(0, len(cross_matrix)):
            for j in range(0, len(cross_matrix[i])):
                if i == j:
                    continue
                cross_points = cross_matrix[i][j]
                sum_check = 0
                for point in cross_points:
                    sum_check += point[2]
                if sum_check == 0:
                    GLI[i][j] /= 100
        
        return GLI, cross_matrix

    def select_obj(self, GLI, objs_GLI, skel_list, z_list, first_skip_line_index):
        if len(objs_GLI) == 0:
            return [0], [0]

        normalize_objs_GLI = min_max_x(objs_GLI)

        len_list, dif_len, correct_length_index, not_correct_length_index = [], [], [], []
        for i, (skel, z) in enumerate(zip(skel_list, z_list)):
            length = len(skel)
            len_list.append(length)
            dif = np.abs(self.full_length - length)
            normalize_dif = dif / self.full_length
            dif_len.append(normalize_dif)
            if length > self.full_length-self.error_length and length < self.full_length + self.error_length:
                correct_length_index.append(i)
            else:
                not_correct_length_index.append(i)
        dif_len = np.array(dif_len)

        ave_z_list = []
        for z, length in zip(z_list, len_list):
            sum_z = np.array(z).sum()
            ave_z = sum_z / length 
            ave_z_list.append(ave_z)
        normalize_ave_z = 1 - min_max_x(ave_z_list)
  
        alpha, beta, gamma = 2, 2, 1

        correct_length_value = []
        for index in correct_length_index:
            correct_length_value.append(alpha*normalize_objs_GLI[index] + gamma*normalize_ave_z[index])
        correct_sort_value = np.argsort(correct_length_value)

        not_correct_length_value = []
        for index in not_correct_length_index:
            result_value = alpha*normalize_objs_GLI[index] + beta*dif_len[index] + gamma*normalize_ave_z[index]
            not_correct_length_value.append(result_value)
        sort_value = np.argsort(not_correct_length_value)

        top_5_index = []
        k = 0
        for k, index in enumerate(correct_sort_value):
            top_5_index.append(correct_length_index[index])
            if k >= 5:
                break

        if k < 4:
            for index in sort_value:
                top_5_index.append(not_correct_length_index[index])
                if len(top_5_index) >= 5:
                    break
        top_5_max_GLI_index = []
        for index in sort_value:
            top_5_max_GLI_index.append(np.argmax(GLI[index]))

        return top_5_index, top_5_max_GLI_index

class Graspability():

    def __init__(self):
        self.TEMPLATE_SIZE = 100
        self.OPEN_WIDTH = cfg["OPEN_WIDTH"]
        self.HAND_THICKNESS_X = 10
        self.HAND_THICKNESS_Y = 25
        self.cutsize = self.TEMPLATE_SIZE/2
        self.interval = 15

    def __main__(self, LC_skel_list, interpolate, region_list, top5_index, top5_maxGLI_index, cross_matrix, labeled_img):
        MI = MakeImage()

        all_region = np.zeros((height, width))
        for region in region_list:
            rimg = MI.make_image(region, 1)
            all_region += rimg

        if top5_index != []:
            count_iterations = 0
            warn_area = 30
            optimal_grasp2 = []

            for obj_index, max_GLI_index in zip(top5_index, top5_maxGLI_index):
                optimal_grasp = self.find_grasp_point(LC_skel_list, obj_index, interpolate, cross_matrix, region_list, all_region, labeled_img)
                if optimal_grasp == []:
                    count_iterations += 1
                    continue
                elif optimal_grasp[0][3][0] < warn_area or optimal_grasp[0][3][0] > height-warn_area or optimal_grasp[0][3][1] < warn_area or optimal_grasp[0][3][1] > width-warn_area:
                    count_iterations += 1
                    print("This grasp position is danger, so next coodinate will be calculate!")
                    continue 
                else:
                    # optimal_grasp2 = self.find_grasp_point2(LC_skel_list, obj_index, max_GLI_index, cross_matrix, region_list, all_region)
                    break

            if count_iterations == 5:
                raise ValueError("Error: 掴むことができる物体がありません")

            return optimal_grasp, optimal_grasp2, obj_index
        else:
            optimal_grasp = self.find_grasp_point(LC_skel_list, 0, [[]], [[]], region_list, all_region, labeled_img)
            return optimal_grasp, [], 0

    def make_template(self):
        L1x = int((self.TEMPLATE_SIZE / 2) - (self.OPEN_WIDTH / 2 + self.HAND_THICKNESS_X))
        L3x = int((self.TEMPLATE_SIZE / 2) - (self.OPEN_WIDTH / 2))
        R1x = int((self.TEMPLATE_SIZE / 2) + (self.OPEN_WIDTH / 2))
        R3x = int((self.TEMPLATE_SIZE / 2) + (self.OPEN_WIDTH / 2 + self.HAND_THICKNESS_X))

        L1y = int((self.TEMPLATE_SIZE / 2) - (self.HAND_THICKNESS_Y / 2))
        L3y = int((self.TEMPLATE_SIZE / 2) + (self.HAND_THICKNESS_Y / 2))
        R1y = int((self.TEMPLATE_SIZE / 2) - (self.HAND_THICKNESS_Y / 2))
        R3y = int((self.TEMPLATE_SIZE / 2) + (self.HAND_THICKNESS_Y / 2))
 
        Hc_original = np.zeros((self.TEMPLATE_SIZE, self.TEMPLATE_SIZE))
        cv2.rectangle(Hc_original, (L1x, L1y), (L3x, L3y), (255, 255, 255), -1)
        cv2.rectangle(Hc_original, (R1x, R1y), (R3x, R3y), (255, 255, 255), -1)

        Hc_original_list = []

        for i in range(4):
            Hc_original = np.zeros((self.TEMPLATE_SIZE, self.TEMPLATE_SIZE))
            cv2.rectangle(Hc_original, (L1x+5*i, L1y), (L3x+5*i, L3y), (255, 255, 255), -1)
            cv2.rectangle(Hc_original, (R1x-5*i, R1y), (R3x-5*i, R3y), (255, 255, 255), -1)
            Hc_original_list.append(Hc_original)

        return Hc_original_list

    def cv2pil(self, image):
        new_image = image.copy()
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_BGRA2RGBA)
        new_image = Image.fromarray(new_image)

        return new_image

    #把持位置決定
    def find_grasp_point(self, skel_list, obj_index, interpolate, cross_matrix, region_list, all_region, labeled_img):
        MI = MakeImage()
        grasp_obj = skel_list[obj_index]                 #obj_indexは把持対象物の番号
        grasp_interpolate = interpolate[obj_index]
        grasp_obj_copy = [tuple(xy) for xy in grasp_obj]
        grasp_interpolate = [tuple(xy) for xy in grasp_interpolate]
        dif_obj = list(set(grasp_obj_copy) - set(grasp_interpolate))
        grasp_region_index = [int(labeled_img[xy[0]][xy[1]]-1) for xy in dif_obj if labeled_img[xy[0]][xy[1]] > 0]

        length = len(grasp_obj)                          #grasp_objの画素数
        obj_interpolate = interpolate[obj_index]         #grasp_objの補間部分
        optimal_grasp = []                               #最適な５つの把持位置がoptimal_graspに保存される
        Hc_original_list = self.make_template()          #グリッパーのテンプレート

        ##############把持対象物が他の物体の下側に存在する場合、その交点を見つける#######################
        underlapped_point = []
        try:
            obj_cross_points = cross_matrix[obj_index]           
        except IndexError:
            obj_cross_points = []                       
        for points in obj_cross_points:                                             
            for point in points:
                if point[2] == 1:
                    index = grasp_obj.index([point[0], point[1]])
                    underlapped_point.append([point[0], point[1], index])
        #################################################################################################

        ##############把持位置探索の範囲を絞る(start_index, finish_index)################################
        underlapped_point.append([grasp_obj[0][0], grasp_obj[0][1], 0])                              #把持対象物の始点の座標を格納
        underlapped_point.append([grasp_obj[length-1][0], grasp_obj[length-1][1], length-1])         #把持対象物の終点の座標を格納
        underlapped_point = np.array(underlapped_point)
        sorted_underlapped_point = underlapped_point[np.argsort(underlapped_point[:, 2])]            #始点からの距離でソート

        point_distance = []
        for i in range(len(sorted_underlapped_point)-1):
            point_distance.append(sorted_underlapped_point[i+1][2] - sorted_underlapped_point[i][2]) #sorted_underlapped_pointに格納されている点と点の距離を求める
        max_underlapped_point_index = np.argmax(point_distance)                                      #点間の距離が最長の部分を見つける
        start_index = sorted_underlapped_point[max_underlapped_point_index][2]                       #該当の点の一つをstart_indexとする
        finish_index = sorted_underlapped_point[max_underlapped_point_index+1][2]                    #該当の点の一つをfinish_indexとする
        if finish_index == length - 1:
            finish_index -= (self.interval + 1)
        #################################################################################################

        ####################把持対象物の画像の作成###############################################################
        if type(grasp_region_index) is list:
            contact_image = np.zeros((height, width))
            for rind in grasp_region_index:
                rimg = MI.make_image(region_list[rind], 1)
                contact_image += rimg
        else:
            contact_image = MI.make_image(region_list[grasp_region_index], 1)
        ################################################################################################

        depth = img_copy.copy()
        depth[all_region==0] *= 0
        # contact_depth = depth.copy()
        # contact_depth = depth * contact_image    #把持対象物のみの深度画像
        # depth[contact_image>0] = 0               #把持対象物の無い深度画像

        ########################対象物に沿って一定間隔で把持位置を求める####################################
        if start_index == 0:
            start_index += self.interval
        prev_poi = [-1, -1]
        
        unique = list(map(list, set(map(tuple, grasp_obj))))
        if not len(unique) == len(grasp_obj):
            print("Warning: 2943 把持対象物の細線座標に重複があります。")
            print("len(unique) = {}, len(grasp_obj) = {}".format(len(unique), len(grasp_obj)))

        for i in range(start_index, finish_index - self.interval, self.interval):
            poi = grasp_obj[i]
            if poi in obj_interpolate:
                continue
            next_poi = grasp_obj[i+self.interval]
            z = depth[poi[0]][poi[1]]
            optimal_grasp = self.graspability(poi, next_poi, prev_poi, optimal_grasp, Hc_original_list, depth, z, i//self.interval, contact_image)
            prev_poi = poi
            # depth = cv2.circle(depth, (poi[1], poi[0]), 1, (255, 0, 0), -1)
        ####################################################################################################

        ##############################################################################
        # vec = np.array(next_poi) - np.array(poi)
        # poi = grasp_obj[-1]
        # next_poi = poi + vec
        # if not poi in obj_interpolate:
        #     optimal_grasp = self.graspability(poi, next_poi, optimal_grasp, Hc_original_list, contact_image, depth)
        #     depth = cv2.circle(depth, (poi[1], poi[0]), 1, (255, 0, 0), -1)
        ##############################################################################

        return optimal_grasp

    def find_grasp_point2(self, skel_list, obj_index, max_GLI_index, cross_matrix, region_list, LC_region_list, all_region):
        MI = MakeImage()
        grasp_obj = skel_list[max_GLI_index]
        grasp_region_index = LC_region_list[obj_index]
        Hc_original_list = self.make_template()
        optimal_grasp = []

        points = cross_matrix[max_GLI_index][obj_index]

        print("Warning: 2973 交点がないときの場合が必要 find_grasp_point2")
        if points == []:
           print("Warning: 交点なし")
           return []

        for point in points:
            if point[2] == 0:
                overlapped_point = [point[0], point[1]]
                overlapped_point_index = grasp_obj.index(overlapped_point)

        try:
            print(overlapped_point_index)
        except UnboundLocalError:
            return []

        if type(grasp_region_index) is list:
            contact_image = np.zeros((height, width))
            for rind in grasp_region_index:
                rimg = MI.make_image(region_list[rind], 1)
                contact_image += rimg
        else:
            contact_image = MI.make_image(region_list[grasp_region_index], 1)

        depth = img_copy.copy()
        depth[all_region==0] *= 0
        # depth[contact_image>0] = 0
        # contact_depth = depth * contact_image

        prev_poi = [-1, -1]
        for i in range(overlapped_point_index - self.interval*2, overlapped_point_index + self.interval*2 + 1, self.interval):
            if i < 0 or i + self.interval >= len(grasp_obj):
                continue
            poi = grasp_obj[i]
            next_poi = grasp_obj[i+self.interval]
            z = depth[poi[0]][poi[1]]
            optimal_grasp = self.graspability(poi, next_poi, prev_poi, optimal_grasp, Hc_original_list, depth, z, i//self.interval, contact_image)
            prev_poi = poi
            # depth = cv2.circle(depth, (poi[1], poi[0]), 1, (255, 0, 0), -1)

        return optimal_grasp

    def graspability(self, poi, next_poi, prev_poi, optimal_grasp, Hc_original_list, depth, z, index, contact_image):
        ###########################################################
        '''
        optimal_grasp(grasp)の構成
        grasp[0]:点の番号
        grasp[1]:方向
        grasp[2]:z座標
        grasp[3]:xy座標
        grasp[4]:グリッパの開き幅
        '''
        ###########################################################
        if z == 0:
            return optimal_grasp

        z -= int(15*(100/max_depth)) + 15
        if z < 0:
            z = 0

        Hc_rotate_list = []
        vector2next = np.array([next_poi[0], next_poi[1]]) - np.array([poi[0], poi[1]])
        if prev_poi[0] > 0:
            vector2prev = np.array([poi[0], poi[1]]) - np.array([prev_poi[0], prev_poi[1]])
            vector2next = (vector2next + vector2prev) // 2
        for Hc_original in Hc_original_list:
            Hc_original = self.cv2pil(Hc_original)
            initial_rotate = math.degrees(math.atan2(vector2next[1], vector2next[0]))
            Hc_rotate_list.append(Hc_original.rotate(initial_rotate))

        contact_Image = Image.fromarray(contact_image)
        collision_per_angle = []
        '''
        collision_per_angleはangleごとに各開き幅での衝突判定をする。
        -1ならすべての開き幅で衝突なし
        0ならすべての開き幅で衝突
        :
        3なら最小開き幅でのみ衝突
        '''       
        # V = Visualize()
        for angle in np.arange(-22.5, 33.75, 22.5):
        # for angle in np.arange(0, 10, 22.5):
            collision_per_angle.append(100)
            for i, Hc_rotate in enumerate(Hc_rotate_list):
                Hc_rotate = Hc_rotate.rotate(angle)
                Hc_rotate = np.array(Hc_rotate)
                CI_crop = np.array(contact_Image.crop((poi[1]-self.cutsize, poi[0]-self.cutsize, poi[1]+self.cutsize, poi[0]+self.cutsize)))
                # CI_crop_color = gray2color(CI_crop)
                # CI_crop_color[Hc_rotate > 0] = [255, 0, 0]
                CI_crop[Hc_rotate == 0] = 0
                if np.sum(CI_crop) > 0:
                    collision_per_angle[-1] = i
                    break
                # V.visualize_1img(CI_crop_color)

        _, Wc = cv2.threshold(depth,z,255,cv2.THRESH_BINARY)
        Wc = self.cv2pil(Wc)
        Wc_crop = Wc.crop((poi[1]-self.cutsize, poi[0]-self.cutsize, poi[1]+self.cutsize, poi[0]+self.cutsize))
        for i, angle in enumerate(np.arange(-22.5, 33.75, 22.5)):
        # for i, angle in enumerate(np.arange(0, 10, 22.5)):
            collision_check = collision_per_angle[i] 
            if collision_check == 0:
                continue
            count = 0
            for j, Hc_rotate in enumerate(Hc_rotate_list):
                if collision_check <= j:
                    continue
                Hc_rotate = Hc_rotate.rotate(angle)
                C = np.array(Wc_crop) * np.array(Hc_rotate)
                C[C>0] = 1

                # print(poi, z)
                # Wc_color = gray2color(np.array(Wc))
                # Wc_color = cv2.copyMakeBorder(Wc_color, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                # Wc_color[poi[0]:poi[0]+100, poi[1]:poi[1]+100][np.array(Hc_rotate) > 0] = [255, 0, 0]
                # Wc_color = Wc_color[50:Wc.height+50, 50:Wc.width+50]
                # Wc_color[contact_image > 0] = [0, 255, 0]
                # Wc_color = cv2.circle(Wc_color, (poi[1], poi[0]), 3, [0, 0, 255], -1)
                # V.visualize_3img(Wc_color, depth, C)

                if np.sum(C) <= 10:
                    count += 1
            if count == collision_check:        
                grasp = []
                grasp.append(index)
                grasp.append(initial_rotate+angle)
                grasp.append(z)
                grasp.append(poi)
                grasp.append(0)
                optimal_grasp.append(grasp)
        
        return optimal_grasp

class GraspPoint():

    def decide_grasp_position(self, optimal_grasp, pcd_matrix_index_image, pcd_matrix):
        best_grasp = optimal_grasp[0]
        x = best_grasp[3][1]
        y = best_grasp[3][0]
        index = int(pcd_matrix_index_image[y][x])
        camera_loc = pcd_matrix[index] / 1000
        calib_path = data_folder_path / "cfg" / "calibmat.txt"
        grasp_position = list(self.transform_camera_to_robot(camera_loc, str(calib_path)))

        obj_orientation = best_grasp[1]
        if obj_orientation+33 > 145+90:
            obj_orientation -= 180
        elif obj_orientation+33 < 145-90:
            obj_orientation += 180 
        yaw = round(obj_orientation, 1)

        grasp_position.append(yaw)
        print("Grasp position is ", grasp_position)

        return grasp_position

    def make_motionfile(self, optimal_grasp, goal_potion, matrix_image, pcd_matrix):
        #motion = [start, end, option, x, y, z, roll, pitch, yaw]
        #optimal_grasp = [graspability, orientation, z, [x, y]]
        motion_list = []
        time, time_interval = 0, 3
        OPEN_WIDTH = cfg["OPEN_WIDTH"]
        max_open_width = cfg["max_open_width"]

        best_grasp = optimal_grasp[0]
        x = best_grasp[3][1]
        y = best_grasp[3][0]

        index = int(matrix_image[y][x])
        camera_loc = pcd_matrix[index]/1000
        calib_path = data_folder_path / "cfg" / "calibmat.txt"
        grasp_point = self.transform_camera_to_robot(camera_loc, calib_path)
        obj_x = grasp_point[0]
        obj_y = grasp_point[1]
        obj_z = grasp_point[2]-0.13
        print("grasp_point = ", grasp_point)
            
        obj_orientation = best_grasp[1]
        if obj_orientation > 180:
            obj_orientation -= 180
        elif obj_orientation < 0:
            obj_orientation += 180 
        obj_roll, obj_pitch, obj_yaw = 0, 0, round(obj_orientation, 1)
        print("grasp_orientation = ", (obj_roll, obj_pitch, obj_yaw))

        open_width = (OPEN_WIDTH-best_grasp[4]*10) / OPEN_WIDTH * max_open_width
        print("open_width = ", open_width)

        goal_x = goal_potion[0]
        goal_y = goal_potion[1]
        goal_z = goal_potion[2]

        upper_z = 0.2

        # motion_list.append([time, time+3, "LHAND_JNT_OPEN"]) #ハンドを開く
        # time += time_interval
        motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", obj_x, obj_y, upper_z, obj_roll, obj_pitch, obj_yaw, open_width]) #対象物の上に移動
        time += time_interval
        motion_list.append([time, time+time_interval+7, "LARM_XYZ_ABS", obj_x, obj_y, obj_z, obj_roll, obj_pitch, obj_yaw, open_width]) #対象物の高さに下ろす
        time += time_interval
        time += 7
        motion_list.append([time, time+time_interval, "LHAND_JNT_CLOSE", 0, 0, 0, 0, 0, 0]) #ハンドを閉じる
        time += time_interval
        motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", obj_x, obj_y, upper_z, obj_roll, obj_pitch, obj_yaw, 0]) #対象物を真上へ持ち上げ
        time += time_interval
        motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", goal_x, goal_y, upper_z, 0, 0, 0, 0]) #対象物を置く地点の上に移動
        time += time_interval
        motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", goal_x, goal_y, goal_z, 0, 0, 0, 0]) #対象物を下ろす
        time += time_interval
        motion_list.append([time, time+time_interval, "LHAND_JNT_OPEN"]) #ハンドを開く
        time += time_interval
        motion_list.append([time, time+time_interval, "LARM_XYZ_ABS", goal_x, goal_y, upper_z, 0, 0, 0, 2.86487]) #ハンドを上げる
        time += time_interval

        with open(data_folder_path / "motionfile" / "motionfile.dat", "w") as f:
            for motion in motion_list:
                length = len(motion)
                for i in range(length):
                    f.write(str(motion[i]))
                    if not i == length - 1:
                        f.write(" ")
                    else:
                        f.write("\n")
        f.close()

        with open(data_folder_path / "motionfile" / "motionfile_csv.csv", "w") as f:
            writer = csv.writer(f)
            for motion in motion_list:
                writer.writerow(motion)
        f.close()

    def transform_camera_to_robot(self, camera_loc, calib_path):
        """
        Transform camera loc to robot loc
        Use 4x4 calibration matrix
        Parameters:
            camera_loc {tuple} -- (cx,cy,cy) at camera coordinate
            calib_path {str} -- calibration matrix file path
        Returns: 
            robot_loc {tuple} -- (rx,ry,rz) at robot coordinate
        """
        # get calibration matrix 4x4
        (cx, cy, cz) = camera_loc
        calibmat = np.loadtxt(calib_path)
        camera_pos = np.array([cx, cy, cz, 1])
        rx, ry, rz, _ = np.dot(calibmat, camera_pos)  # unit: mm --> m
        return (rx, ry, rz)

def show(combi, side, line_list):
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

def main(input_image, pcd_matrix_index_image, pcd_matrix, MaxDepth):
    start = time.time()

    MI = MakeImage()
    V = Visualize()
    SI = SaveImage()

    global height, width, img_copy, max_depth
    height, width = input_image.shape
    img_copy = input_image.copy()
    max_depth = MaxDepth
    cv2.imwrite(str(save_folder_path / "depthimg.png"), input_image)

    rg_time = time.time()
    ##################################セグメンテーション############################################
    RG = RegionGrowing(input_image)
    region_list2, labeled_img = RG.search_seed()
    SI.save_region(region_list2, "RG")
    print("Segmentation is succeeded!")
    ################################################################################################
    rg_elapsed_time = time.time() - rg_time
    print("Time costs for RegionGrowing is ", rg_elapsed_time)

    # V.visualize_region(region_list2)

    sk_time = time.time()
    ##################################細線化########################################################
    Skel = Skeletonize()
    S = Sort()
    skel_list, branch_point_list, ad_branch_point_list, end_point_list, skip_region_index, skel2region_index, last_region_index = Skel.skeletonize_region_list(region_list2)
    sorted_skel_list = S.sort_skel_list(skel_list, branch_point_list, ad_branch_point_list, end_point_list)
    sorted_skel_list, skel2region_index, first_skip_skel, first_skip_skel2region_index = S.first_check(sorted_skel_list, skel2region_index)
    sorted_skel_list, skel2region_index = sunder_curve(sorted_skel_list, skel2region_index)
    SI.save_region(sorted_skel_list, "skeltonized")
    print("Skeletonize is succeeded!")
    ################################################################################################
    sk_elapsed_time = time.time() - sk_time
    print("Time costs for Skeletonize is ", sk_elapsed_time)

    cl_time = time.time()
    ##################################細線接続######################################################
    CL = ConnectLine()
    CL_skel_list, CL_z_list, CL_interpolate, CL_index_list, CL_connect_ep = CL.__main__(sorted_skel_list, end_point_list)
    correct_skel_index, correct_CL_index, connect_skel_index, connect_ep = CL.connect_check(CL_skel_list, CL_index_list, CL_connect_ep)
    skip_region_index, last_skel_index = CL.which_region_left(skip_region_index, skel2region_index, first_skip_skel2region_index, last_region_index, first_skip_skel, correct_skel_index)
    
    # print(skip_region_index)
    # print(last_skel_index)
    # aimg = np.zeros((height, width))
    # for index in last_skel_index:
    #     simg = MI.make_image(sorted_skel_list[index], 1)
    #     aimg += simg
    # V.visualize_1img(aimg)
    # rimg = np.zeros((height, width))
    # for index in skip_region_index:
    #     print(index)
    #     rimg += MI.make_image(region_list2[index], 1)
    # V.visualize_1img(rimg)
    # input()
    
    SI.save_region(CL_skel_list, "connected_by_skel")
    print("Connect line process is succeeded!")
    ################################################################################################
    cl_elapsed_time = time.time() - cl_time
    print("Time costs for ConnectLine is ", cl_elapsed_time)

    cr_time = time.time()
    ##################################領域接続######################################################
    CP = CenterPoint()
    center_points_list, cp2region_index = CP.search_center_points(region_list2, skip_region_index)
    CR = ConnectRegion()
    new_center_points_list, line_list, cp2skel_index = CR.point2line(center_points_list, last_skel_index, sorted_skel_list, cp2region_index, skel2region_index)
    SI.save_region(line_list, "centerpoint2line")
    thetas, ends, depthes = CR.end_point_information(line_list)
    connect_ep_index, connect_cp_index, line_list, ends, thetas, depthes = CR.match_index(ends, connect_skel_index, connect_ep, cp2skel_index, sorted_skel_list, line_list, thetas, depthes)
    combi, side, pos, center_points_list = CR.connect_region(new_center_points_list, line_list, thetas, ends, depthes, connect_cp_index, connect_ep_index)
    # show(combi, side, pos, line_list)
    CR_line_list, CR_interpolate, CR_z_list, CR_index_list = CR.check_connect(combi, side, pos, line_list)
    SI.save_region(CR_line_list, "connected_by_region")
    SI.save_centerpoints(center_points_list)
    line_list, interpolate, z_list, first_skip_line_index = CR.merge_CR_and_CL(CR_line_list, CR_interpolate, CR_z_list, correct_CL_index, CL_skel_list, CL_interpolate, CL_z_list, first_skip_skel, connect_cp_index, combi, line_list) 
    SI.save_region(line_list, "connected")
    print("Center point process is succeeded!")
    ################################################################################################
    cr_elapsed_time = time.time() - cr_time
    print("Time costs for ConnectRegion is ", cr_elapsed_time)

    gli_time = time.time()
    ################################GLIの計算#######################################################
    GLI = GaussLinkingIintegral()
    GLI_matrix, objs_GLI, cross_matrix = GLI.calculate_GLI(line_list, z_list, interpolate)
    top5_index, top5_maxGLI_index = GLI.select_obj(GLI_matrix, objs_GLI, line_list, z_list, first_skip_line_index)
    print("GLI caluculation is succeeded!")
    ################################################################################################
    gli_elapsed_time = time.time() - gli_time
    print("Time costs for GLI is ", gli_elapsed_time)

    ga_time = time.time()
    ###############################把持位置の取得###################################################
    GA = Graspability()
    optimal_grasp, optimal_grasp2, obj_index = GA.__main__(line_list, interpolate, region_list2, top5_index, top5_maxGLI_index, cross_matrix, labeled_img)
    # print("optimal_grasp = {}".format(optimal_grasp))
    # print("opitmal_grasp2 = {}".format(optimal_grasp2))
    SI.save_grasp_position(line_list, obj_index, optimal_grasp)
    print("Decision of grasp position is succeeded!")
    ################################################################################################
    ga_elapsed_time = time.time() - ga_time
    print("Time costs for Graspability is ", ga_elapsed_time)

    gp_time = time.time()
    ##############################モーションファイルの作成##########################################
    GP = GraspPoint()
    # goal_position = [0.4, 0.38, 0]
    # GP.make_motionfile(optimal_grasp, goal_position, pcd_matrix_index_image, pcd_matrix)
    grasp_position = GP.decide_grasp_position(optimal_grasp, pcd_matrix_index_image, pcd_matrix)
    print("Decision of grasp position is succeeded!")
    ################################################################################################
    gp_elapsed_time = time.time() - gp_time
    print("Time costs for GraspPoint is ", gp_elapsed_time)

    elapsed_time = time.time() - start#処理の終了時間を取得
    print("Run time costs to decide grasp position is {}\n".format(elapsed_time))

    return grasp_position

if __name__ == "__main__":
    #画像の読み込み
    import prePly2depth
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(str(ply_filepath))
    pc = np.asanyarray(pcd.points)
    input_image, matrix_image, pcd_matrix, MaxDepth = prePly2depth.main(pc)
    # import check
    # input_image, matrix_image, pcd_matrix, MaxDepth, gray_img = check.main(ply_filepath)
    main(input_image, matrix_image, pcd_matrix, MaxDepth)