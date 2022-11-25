# -*- coding: utf-8 -*-
from __future__ import print_function
from multiprocessing.sharedctypes import Value
import os
import shutil
import argparse
import time
import sys
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np
import time
import math
from datetime import datetime as dt
import prePly2depth
import open3d as o3d
import pathlib

#LOG_MODEを1にするとM0,M1,M2のどれを実行しても、他２つのMでの推定値も記録しておく※M:Mehod
LOG_MODE = 0
# Method2のモード設定
# 大域領域(５００×５００画像を１２５×１２５に圧縮)で探索する場合は１
GLOBAL_OR_LOCAL = 1
# 最終的は把持位置を[Graspability]と[1-ひっかかり予測値]の調和平均で決定するなら１、閾値判定の場合は０に設定し、適切な閾値を該当箇所(1445行目あたり)で設定
Harmonic_OR_Threshold = 1

#最終的に求まった"TOP_NUMBER"個の把持位置情報を格納
#"Top_○○"は同じインデックスが配列名の値を表す。
Top_Graspability = np.array([])
Top_ImgX = np.array([])
Top_ImgY = np.array([])
Top_ImgZ = np.array([])#0〜255の輝度値
Top_Angle = np.array([])
Top_CountA = np.array([])
Top_CountB = np.array([])

def Graspability(METHOD, tstr, img_pass):#//おそらくMETHODは実験番号、tstrは日にち　

    #モーションファイルがあれば２回目以降と判断し、テンプレートの生成はスキップ
    #=== 2回目以降の処理
    # if(os.path.exists("./MOTION/motion.txt")):
    #     print("REMOVED MOTIONFILE")
    #     os.remove("./MOTION/motion.txt")
    #=== 準備（ハンドのテンプレートを生成）=================================================================
    #else:
    #グレースケールで読み込み
    TEMPLATE_SIZE = 100
    black = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE))
    cv2.imwrite(str(save_folder_path/"black250.png"), black)
    #//○tは接触領域画像、○cは衝突領域画像
    Ht_original = cv2.imread(str(save_folder_path/"black250.png"), 0)
    Hc_original = cv2.imread(str(save_folder_path/"black250.png"), 0)
    
    HAND_THICKNESS_X = 2#シミュレーション時は2だが、実際のハンドに合わせて15mm//(おそらく横幅)
    HAND_THICKNESS_Y = 5#シミュレーション時は5だが、実際のハンドに合わせて25mm//(おそらく縦幅)
    BEFORE_TO_AFTER = 5#//おそらく奥行き
    OPEN_WIDTH = 50

    #       L1  L2              R1  R2
    #       L4  L3              R4  R3

    L1x = int((TEMPLATE_SIZE / 2) - (OPEN_WIDTH / 2 + HAND_THICKNESS_X))
    L2x = int((TEMPLATE_SIZE / 2) - (OPEN_WIDTH / 2))
    L3x = int((TEMPLATE_SIZE / 2) - (OPEN_WIDTH / 2))
    R1x = int((TEMPLATE_SIZE / 2) + (OPEN_WIDTH / 2))
    R3x = int((TEMPLATE_SIZE / 2) + (OPEN_WIDTH / 2 + HAND_THICKNESS_X))
    R4x = int((TEMPLATE_SIZE / 2) + (OPEN_WIDTH / 2))

    L1y = int((TEMPLATE_SIZE / 2) - (HAND_THICKNESS_Y / 2))
    L2y = int((TEMPLATE_SIZE / 2) - (HAND_THICKNESS_Y / 2))
    L3y = int((TEMPLATE_SIZE / 2) + (HAND_THICKNESS_Y / 2))
    R1y = int((TEMPLATE_SIZE / 2) - (HAND_THICKNESS_Y / 2))
    R3y = int((TEMPLATE_SIZE / 2) + (HAND_THICKNESS_Y / 2))
    R4y = int((TEMPLATE_SIZE / 2) + (HAND_THICKNESS_Y / 2))

    cv2.rectangle(Hc_original, (0, 0), (250, 250), (0, 0, 0), -1)
    cv2.rectangle(Ht_original, (0, 0), (250, 250), (0, 0, 0), -1)
    #左上の角と右下の角を指定すると四角形が描かれる
    cv2.rectangle(Hc_original, (L1x, L1y), (L3x, L3y), (255, 255, 255), -1)
    cv2.rectangle(Hc_original, (R1x, R1y), (R3x, R3y), (255, 255, 255), -1)
    cv2.rectangle(Ht_original, (L2x, L2y), (R4x, R4y), (255, 255, 255), -1)

    #os.makedirs("./tmp")
    cv2.imwrite(str(save_folder_path/'tmp/Ht_original.png'), Ht_original)
    cv2.imwrite(str(save_folder_path/'tmp/Hc_original.png'), Hc_original)
    # ここまで準備 ==============================================================================
    HandRotationStep = 22.5
    HandDepthStep = 10 #25でも可能
    GripperD = 25#//おそらくグリッパーのZ方向の大きさ
    InitialHandDepth = 100#//初期のハンドの深さ(Z方向)
    FinalHandDepth = 251
    GaussianKernelSize = 75
    GaussianSigma = 25
    CountA = 0#//深さパターン数のカウント
    CountB = 0#//回転パターン数のカウント
    #"all_○○"配列を初期化
    #all_LabelCenter_Area = np.array([])
    all_LabelCenter_X = np.array([])
    all_LabelCenter_Y = np.array([])
    all_LabelCenter_CountA = np.array([])
    all_LabelCenter_CountB = np.array([])
    all_LabelCenter_Graspability = np.array([])

    #深さパターン数だけマップ生成
    for HandDepth in range(InitialHandDepth, FinalHandDepth, HandDepthStep):#//InitialHandDepthからFinalHandDepthまでHandDepthStepごとにHandDepthに代入(0~201で50ずつHandDepthに代入)
        CountA += 1
        #Depth_original = cv2.imread("./DEPTH_TMP/depth.bmp",0)
        Depth_original = cv2.imread("{}".format(img_pass), 0)
        _, Wt = cv2.threshold(Depth_original, HandDepth + GripperD, 255, cv2.THRESH_BINARY)#//高さHandDepth+GripperDを基準に二値化する(これより高ければ白)
        #//_, Wtはcv2.thresholdの２つめの返数をWtに代入し、一つ目の返数を捨てている
        _, Wc = cv2.threshold(Depth_original, HandDepth, 255, cv2.THRESH_BINARY)#//高さHandDepthを基準に二値化する

        cv2.imwrite(str(save_folder_path/"tmp/Wt{}.png".format(CountA)), Wt)#//深さパターンマップの保存
        cv2.imwrite(str(save_folder_path/"tmp/Wc{}.png".format(CountA)), Wc)

        CountB = 0
        #回転パターン数だけマップ生成
        for HandRotation in np.arange(0, 180, HandRotationStep):#//0~180でHandRotationStepごとにHandRotationに代入
            CountB += 1
            if CountA == 1:#//深さが初期状態の場合、元の画像をそのまま使う(というよりcountA==1のものしか使っていない) 
                Ht = Image.open(save_folder_path/"tmp/Ht_original.png")
                Ht = Ht.rotate(HandRotation)#//回転させる
                Ht.save(save_folder_path/"tmp/Ht{}.png".format(CountB))#//回転パターンマップの保存
                Hc = Image.open(save_folder_path/"tmp/Hc_original.png")
                Hc = Hc.rotate(HandRotation)
                Hc.save(save_folder_path/"tmp/Hc{}.png".format(CountB))
            '''
            else:
                Ht = Image.open("./BEST/{}/tmp/Ht{}.png".format(CountB))
                Hc = Image.open("./BEST/{}/tmp/Hc{}.png".format(CountB))
            '''

            #Wt = cv2.imread("./tmp/Wt{}.png".format(CountA),0)
            #Wc = cv2.imread("./tmp/Wc{}.png".format(CountA),0)

            #ハンドモデルの接触領域計算
            Ht = np.array(Image.open(save_folder_path/"tmp/Ht{}.png".format(CountB)).convert('L'))#//回転パターンマップを8bitのグレースケールに変換
            T = cv2.filter2D(Wt, -1, Ht)#//HtをカーネルフィルタとしてWtへの畳み込みを計算する(つまりWtとHtの類似性を求めているらしい)
            '''
            if CountA == 1:
                np.set_printoptions(threshold=np.inf)
                fp = open('./MOTION/Wt.txt', 'wt')
                print("{}".format(Wt), file=fp)
                fp.close()
                fp = open('./MOTION/Ht.txt', 'wt')
                print("{}".format(Ht), file=fp)
                fp.close()
                fp = open('./MOTION/T.txt', 'wt')
                print("{}".format(T), file=fp)
                fp.close()  
            '''
            cv2.imwrite(str(save_folder_path/"tmp/T{}_{}.png".format(CountA, CountB)), T)#//接触領域の保存

            #ハンドモデルの衝突領域計算
            Hc = np.array(Image.open(save_folder_path/"tmp/Hc{}.png".format(CountB)).convert('L'))
            C = cv2.filter2D(Wc, -1, Hc)
            cv2.imwrite(str(save_folder_path/"tmp/C{}_{}.png".format(CountA, CountB)), C)

            #ハンドモデルの衝突領域を反転（補集合）
            Cbar = 255 - C
            cv2.imwrite(str(save_folder_path/"tmp/Cbar{}_{}.png".format(CountA, CountB)), Cbar)

            #Graspabilityマップを作成（ぼかし前）
            T_and_Cbar = T & Cbar#//共通部分の作成
            cv2.imwrite(str(save_folder_path/"tmp/T_and_Cbar{}_{}.png".format(CountA, CountB)), T_and_Cbar)

            #Graspabilityマップを完成（ぼかし後）
            G = cv2.GaussianBlur(T_and_Cbar, (GaussianKernelSize, GaussianKernelSize), GaussianSigma, GaussianSigma)#//ガウシアンフィルタをかける(白い部分からの距離に応じた平滑化)
            #//GaussianBlur(img,(カーネルサイズX,カーネルサイズY),標準偏差値X,標準偏差値Y)(カーネルサイズは近傍何画素に注目するか、標準偏差は注目画素の距離に応じた重みをどれくらいにするかを決める)
            cv2.imwrite(str(save_folder_path/"tmp/G{}_{}.png".format(CountA, CountB)), G)

            '''
            src = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/T_and_Cbar{}_{}.png".format(METHOD, tstr, CountA, CountB), 0)
            #この画像は0か255の輝度値しか持たないが、安全のためしきい値を122に設定
            ret, thresh = cv2.threshold(src, 122, 255, cv2.THRESH_BINARY)
            T_and_Cbar_Labeled = cv2.connectedComponentsWithStats(thresh)
            '''

            #ラベリング =========================================================================
            src = cv2.imread(str(save_folder_path/"tmp/T_and_Cbar{}_{}.png".format(CountA, CountB)), 0)#//共通部分の画像を二値化して出力
            #ラベリングするには２値化処理が必須。122を閾値として「0or255の画像」→「バイナリ画像」に変換
            ret, thresh = cv2.threshold(src, 122, 255, cv2.THRESH_BINARY)
            T_and_Cbar_Labeled = cv2.connectedComponentsWithStats(thresh)#//二値画像の領域ラベル分け(T_and_Cbar_Labeled[0]にはラベル数、T_and_Cbar_Labeled[1]はラベリングした行列(画像)、T_and_Cbar_Labeled[2]はオブジェクトのバウンディングボックスとサイズ、T_and_Cbar_Labeled[3]はオブジェクトの重心)
            #必要な情報だけ抽出し、
            data = np.delete(T_and_Cbar_Labeled[2], 0, 0)#２次元配列data = [[ラベル１のバウンディングボックス左上X, ラベル１のバウンディングボックス左上Y, ラベル１のバウンディングボックスX幅, ラベル１のバウンディングボックスY幅, ラベル１の面積], ...]
            #//バウンディングボックスとはオブジェクトを囲む枠のこと(拡大縮小のときに対象物を囲むやつみたいなもの)
            #//T_and_Cbar_Labeled[2]の0行目(配列の関係で0行目から始まる)はラベル0の情報で、ラベル0は二値化画像の黒い部分。つまり、邪魔な部分なので消す
            center = np.delete(T_and_Cbar_Labeled[3], 0, 0)#２次元配列center = [[ラベル１の中心x, ラベル１の中心y], ...]
            #本当に必要な情報だけさらに抽出
            #LabelCenter_Area = data[:,4]#1次元配列all_LabelCenter_Area = [ラベル１の面積, ラベル２の面積, ...]
            LabelCenter_X = center[:, 0]#1次元配列all_LabelCenter_X = [ラベル１の中心X, ラベル２の中心X, ...]#//配列centerの0列目を抽出
            LabelCenter_Y = center[:, 1]#1次元配列all_LabelCenter_Y = [ラベル１の中心Y, ラベル２の中心Y, ...]

            # print("ラベル数:",T_and_Cbar_Labeled[0]-1)

            #ラベル数分の"CountA"の値を格納
            _tmpA = np.zeros(LabelCenter_X.shape[0], dtype=int)
            #//○.shape[0]で何行あるかを調べる(○.shape[1]では何列あるか)
            #//np.zeros(大きさ、型)ですべての要素が0の配列の生成
            _tmpA.fill(CountA)
            #//○.fill()はカッコ内の数字で配列を埋める
            #print(_tmpA)
            #ラベル数分の"CountB"の値を格納
            _tmpB = np.zeros(LabelCenter_X.shape[0], dtype=int)
            _tmpB.fill(CountB)
            #print(_tmpB)

            #"all_○○"は同じインデックスが配列名の値を表す。
            #all_LabelCenter_Area = np.append(all_LabelCenter_Area,LabelCenter_Area)
            all_LabelCenter_X = np.append(all_LabelCenter_X, LabelCenter_X)
            #//np.append(a,b)は配列aの末尾に配列bを付け加える(つまり、all_LabelCenter_Xには生成されるすべてのLabelCenter_Xが含まれている)
            all_LabelCenter_Y = np.append(all_LabelCenter_Y, LabelCenter_Y)
            all_LabelCenter_CountA = np.append(all_LabelCenter_CountA, _tmpA)
            all_LabelCenter_CountB = np.append(all_LabelCenter_CountB, _tmpB)
            #Graspabilityは個別に求める必要がある
            for i in range(LabelCenter_X.shape[0]):
                all_LabelCenter_Graspability = np.append(all_LabelCenter_Graspability, G[int(LabelCenter_Y[i])][int(LabelCenter_X[i])])#//ぼかした後の画像からラベル分けされた領域の中心の輝度(値)を格納している
            #ここまでラベリング関連処理 ===================================================================
    #=============================　ここまでGraspabilityマップ生成ステップ　　==============================
    #=============================　ここからGraspability最大位置探索ステップ　==============================
    #Pythonの大域変数は"global"をつけないと操作できない
    global Top_Graspability
    global Top_ImgX
    global Top_ImgY
    global Top_ImgZ
    global Top_Angle
    global Top_CountA
    global Top_CountB

    sorted_index_of_Graspability = np.argsort(all_LabelCenter_Graspability)[::-1]#//各列に対して降順ソートし、添字を返す
    # for i in range(all_LabelCenter_Area.shape[0]):
    #     #index = int( sorted_index_of_Area[_cnt] )
    #     #_Area = all_LabelCenter_Area[index]
    #     _x = int(all_LabelCenter_X[i])
    #     _y = int(all_LabelCenter_Y[i])
    #     _CountA = int(all_LabelCenter_CountA[i])
    #     _CountB = int(all_LabelCenter_CountB[i])
    #     _GraspabilityMap = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/G{}_{}.png".format(METHOD,tstr,_CountA,_CountB),0)
    #     all_LabelCenter_Graspability = np.append(all_LabelCenter_Graspability,_GraspabilityMap[_y][_x])

    #把持位置が近いものはスキップ
    _threshold_distance = 50
    #print (all_LabelCenter_Area.shape[0])
    _cnt = 0 #_cntは候補がTOP_NUMBER個見つかるまで（top_countがTOP_NUMBERと一致するまで）カウントアップし続ける//実際に利用される配列の添字
    top_count = 0 #これがTOP_NUMBERになるまで、または、Areaを持つ候補がなくなるまで続ける//距離が近いものを除いた実質的な候補(フラグとしての働き):
    while (top_count < TOP_NUMBER and _cnt < all_LabelCenter_Graspability.shape[0]):
        #第"top_count"番目に面積の大きいラベルについて
        index = int(sorted_index_of_Graspability[_cnt])#//注目する座標の添字を求める
        #_Area = all_LabelCenter_Graspability[index]
        _x = int(all_LabelCenter_X[index])
        _y = int(all_LabelCenter_Y[index])
        _CountA = int(all_LabelCenter_CountA[index])
        _CountB = int(all_LabelCenter_CountB[index])
        _Graspability = all_LabelCenter_Graspability[index]
        #_GraspabilityMap = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/G{}_{}.png".format(METHOD,tstr,_CountA,_CountB),0)

        #端っこ過ぎる把持位置（上下左右から"DismissAreaWidth_Graspability"ピクセル以内）は除外
        if (DismissAreaWidth_Graspability < _x) and (_x < TRIMMED_WIDTH-DismissAreaWidth_Graspability) and (DismissAreaWidth_Graspability < _y) and (_y < TRIMMED_HEIGHT-DismissAreaWidth_Graspability):
            if top_count == 0:
                Top_Graspability = np.append(Top_Graspability, _Graspability)
                Top_ImgX = np.append(Top_ImgX, _x)
                Top_ImgY = np.append(Top_ImgY, _y)
                Top_ImgZ = np.append(Top_ImgZ, int(InitialHandDepth + HandDepthStep*(_CountA-1) + GripperD/2))
                #回転角はラジアンで保存
                Top_Angle = np.append(Top_Angle, HandRotationStep*(_CountB-1)*(math.pi)/180)
                #ログ用にCountA,CountBもとっておく
                Top_CountA = np.append(Top_CountA, _CountA)
                Top_CountB = np.append(Top_CountB, _CountB)
                top_count += 1
            else:
                for i in range(top_count):
                    #既存上位の候補に、把持位置が近いものは候補外//もう候補に選ばれたもの全てと比較し一つでも近いものがあればアウト
                    if (_x - Top_ImgX[i])*(_x - Top_ImgX[i])+(_y - Top_ImgY[i])*(_y - Top_ImgY[i]) < _threshold_distance*_threshold_distance:#//距離の2乗を比較、近ければbreak
                        break
                    elif i == top_count - 1:#//すべての候補に近くなければここに入る
                        Top_Graspability = np.append(Top_Graspability, _Graspability)
                        Top_ImgX = np.append(Top_ImgX, _x)
                        Top_ImgY = np.append(Top_ImgY, _y)
                        Top_ImgZ = np.append(Top_ImgZ, int(InitialHandDepth + HandDepthStep*(_CountA-1) + GripperD/2))
                        #回転角はラジアンで保存
                        Top_Angle = np.append(Top_Angle, HandRotationStep*(_CountB-1)*(math.pi)/180)
                        #ログ用にCountA,CountBもとっておく
                        Top_CountA = np.append(Top_CountA, _CountA)
                        Top_CountB = np.append(Top_CountB, _CountB)
                        top_count += 1
        _cnt += 1
    #もしTOP_NUMBER個見つからなかった場合は距離を気にせずGraspabilityの大きい順のデータで残りを埋める
    if top_count < TOP_NUMBER:
        print("探索失敗（距離を無視してGraspabilityの大きい順で代用します）")
        #一旦リセット
        Top_Graspability = np.array([])
        Top_ImgX = np.array([])
        Top_ImgY = np.array([])
        Top_ImgZ = np.array([])#0〜255の輝度値
        Top_Angle = np.array([])
        Top_CountA = np.array([])
        Top_CountB = np.array([])
        i = 0
        _detected = 0
        while _detected < TOP_NUMBER:
            index = int(sorted_index_of_Graspability[i])
            #_Area = all_LabelCenter_Area[index]
            _x = int(all_LabelCenter_X[index])
            _y = int(all_LabelCenter_Y[index])

            if (DismissAreaWidth_Graspability < _x) and (_x < TRIMMED_WIDTH-DismissAreaWidth_Graspability) and (DismissAreaWidth_Graspability < _y) and (_y < TRIMMED_HEIGHT-DismissAreaWidth_Graspability):
                _CountA = int(all_LabelCenter_CountA[index])
                _CountB = int(all_LabelCenter_CountB[index])
                _Graspability = int(all_LabelCenter_Graspability[index])
                #_GraspabilityMap = cv2.imread("./Method{}/EXPERIMENT/{}/tmp/G{}_{}.png".format(METHOD,tstr,_CountA,_CountB),0)
                Top_Graspability = np.append(Top_Graspability, _Graspability)
                Top_ImgX = np.append(Top_ImgX, _x)
                Top_ImgY = np.append(Top_ImgY, _y)
                Top_ImgZ = np.append(Top_ImgZ, int(InitialHandDepth + HandDepthStep*(_CountA-1) + GripperD/2))
                #回転角はラジアンで保存
                Top_Angle = np.append(Top_Angle, HandRotationStep*(_CountB-1)*(math.pi)/180)
                #ログ用にCountA,CountBもとっておく
                Top_CountA = np.append(Top_CountA, _CountA)
                Top_CountB = np.append(Top_CountB, _CountB)
                _detected += 1 #探索範囲内のやつ1個見つけた
            i += 1 #ループ続行
            print("i={}".format(i))

 #Graspabilityの高い把持位置"TOP_NUMBER"箇所を１枚に描画
#１位.赤色、２位以降.黄色
def draw_best_hand_with_work(best_pass, B, G, R):
    img = cv2.imread('{}'.format(best_pass), 1)  #※グレーで読み込み（0）すると，円や直線も白か黒しか描けなくなるので注意
    #
    for top_count in range(TOP_NUMBER):
        ImgX = Top_ImgX[top_count]
        ImgY = Top_ImgY[top_count]
        Angle = Top_Angle[top_count]
        #finger_topは画像上で上側の指で，finger_bottomは画像上で下側の指
        finger_top_x = int(ImgX + (25 * math.cos(Angle)))
        finger_top_y = int(ImgY - (25 * math.sin(Angle)))
        finger_bottom_x = int(ImgX - (25 * math.cos(Angle)))
        finger_bottom_y = int(ImgY + (25 * math.sin(Angle)))
        #ハンドをカラーで描画
        if top_count == 0:
            cv2.circle(img, (finger_top_x, finger_top_y), 5, (0, 0, 255), -1)
            cv2.circle(img, (finger_bottom_x, finger_bottom_y), 5, (0, 0, 255), -1)
            cv2.line(img, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (0, 0, 255), 3)
        else:
            cv2.circle(img, (finger_top_x, finger_top_y), 5, (int(B*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(G*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(R*(TOP_NUMBER-top_count)/(TOP_NUMBER-1))), -1)
            cv2.circle(img, (finger_bottom_x, finger_bottom_y), 5, (int(B*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(G*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(R*(TOP_NUMBER-top_count)/(TOP_NUMBER-1))), -1)
            cv2.line(img, (finger_top_x, finger_top_y), (finger_bottom_x, finger_bottom_y), (int(B*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(G*(TOP_NUMBER-top_count)/(TOP_NUMBER-1)), int(R*(TOP_NUMBER-top_count)/(TOP_NUMBER-1))), 3)
    #保存
    cv2.imwrite('{}'.format(best_pass), img)

# モーションファイルを生成する関数
def generate_motion(center_x, center_y, center_z, angle):
#2次元キャリブレーション（内側底面）==============================
    img = Image.open('{}'.format(img_pass))  #※グレーで読み込み
    #pixels = img.load()
    # #把持中心位置を中心とする矩形領域の最大輝度値を取得
    MaxDepthPixelValue = 0
    for x in range(int(center_x-CATCH_DEPTH_WINDOW/2), int(center_x+CATCH_DEPTH_WINDOW/2)):
         for y in range(int(center_y-CATCH_DEPTH_WINDOW/2), int(center_y+CATCH_DEPTH_WINDOW/2)):
             if img.getpixel((x, y)) > MaxDepthPixelValue:
                 MaxDepthPixelValue = img.getpixel((x, y))
    print('把持中心位置近傍の最大輝度値：{}\n'.format(MaxDepthPixelValue))
    DistanceFromYCAM = MAX_DISTANCE - ((float(MaxDepthPixelValue) / 255.0) * (MAX_DISTANCE - MIN_DISTANCE)) #単位メートル
    print("DistanceFromYCAM={}".format(DistanceFromYCAM))
    TotalZ = MAX_DISTANCE + 0.165
    #RealZ = TotalZ - DistanceFromYCAM - 0.015 # -0.015は補正項 ※単位メートル
    # Work1,2,3,4用 =====
    RealZ = TotalZ - DistanceFromYCAM -0.023#は補正項 ※単位メートル
    # ネジ用 ※他のワークより太いので =====
    #RealZ = TotalZ - DistanceFromYCAM -0.02#は補正項 ※単位メートル

    RealX = StandardPointX - (BoxX * (float(center_y) / float(TRIMMED_HEIGHT)))
    RealY = StandardPointY - (BoxY * (float(center_x) / float(TRIMMED_WIDTH))) #-0.005#補正項(キャリブレーションがズレるとき限定）
#===================================================================================
#３次元キャリブレーション ===============================================================
#    img=Image.open('{}'.format(img_pass))  #※グレーで読み込み
#    #pixels = img.load()
#    # #把持中心位置を中心とする矩形領域の最大輝度値を取得
#    MaxDepthPixelValue = 0
#    for x in range(int(center_x-CATCH_DEPTH_WINDOW/2),int(center_x+CATCH_DEPTH_WINDOW/2)):
#         for y in range(int(center_y-CATCH_DEPTH_WINDOW/2),int(center_y+CATCH_DEPTH_WINDOW/2)):
#             if(img.getpixel((x,y)) > MaxDepthPixelValue):
#                 MaxDepthPixelValue = img.getpixel((x,y))
#    print('把持中心位置近傍の最大輝度値：{}\n'.format(MaxDepthPixelValue))
#    DistanceFromYCAM = MAX_DISTANCE - ( ( float(MaxDepthPixelValue) / 255.0 ) * ( MAX_DISTANCE - MIN_DISTANCE ) ) #単位メートル
#    print("DistanceFromYCAM={}".format(DistanceFromYCAM))
#    #「キャリブレーション行列＋画像上のx,y＋世界座標のZ」を元に、世界座標のX,Yを導出
#    c = np.load("./Calibration/CalibMat.npy")
#    A = np.array([[c[0][0]-c[8][0]*(center_x+LEFT_MARGIN), c[1][0]-c[9][0]*(center_x+LEFT_MARGIN)]])
#    A = np.insert(A,1,[c[4][0]-c[8][0]*(center_y+TOP_MARGIN), c[5][0]-c[9][0]*(center_y+TOP_MARGIN)],axis=0)
#    b = np.array([[(center_x+LEFT_MARGIN)-c[3][0]-(c[2][0]-c[10][0]*(center_x+LEFT_MARGIN))*DistanceFromYCAM]])
#    b = np.insert(b,1,[(center_y+TOP_MARGIN)-c[7][0]-(c[6][0]-c[10][0]*(center_y+TOP_MARGIN))*DistanceFromYCAM],axis=0)
#    A_inv = np.linalg.inv(A)
#    XY = np.dot(A_inv,b)
#    RealX = XY[0][0]
#    RealY = XY[1][0]
#    #「Astraからの距離」と「Nextage座標のZ」が常に一定値　（TotalZ）
#    #TotalZ= YCAMから底面までの距離+底面ピッキング時のZ座標（0.16） ※単位メートル
#    TotalZ = MAX_DISTANCE + 0.160
#    #RealZ = TotalZ - DistanceFromYCAM - 0.015 # -0.015は補正項 ※単位メートル
#    RealZ = TotalZ - DistanceFromYCAM -0.008#は補正項 ※単位メートル
#    #RealZ = 0.915 - DistanceFromAstra
#======================================================================================
    print("X座標：{}\nY座標：{}\nZ座標：{}\n".format(RealX, RealY, RealZ))

    if RealZ >= 0.22:
        print("Error: Z座標が上限以上（{}）なので0.22にします".format(RealZ))
        RealZ = 0.22
    elif RealZ <= 0.165:
        print("Error: Z座標が下限以下（{}）なので0.16にします".format(RealZ))
        RealZ = 0.165

    #XとYについても一応確認
    if RealX <= StandardPointX - BoxX or StandardPointX <= RealX:
        print("Error: X座標が範囲外です")
    if RealY <= StandardPointY - BoxY or StandardPointY <= RealY:
        print("Error: Y座標が範囲外です")

    #回転角を導出
    NxAngle = 180.0 * angle / math.pi
    #-90度〜90度の範囲ならすでにOK
    if NxAngle < -90:
        NxAngle = 180 + NxAngle
    elif 90 < NxAngle:
        NxAngle = NxAngle - 180
    print("angle = {} ".format(NxAngle))

    #モーションファイルを生成(左手使用Version)
    fp = open(save_folder_path/'motion.txt', 'wt')
    print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 0 -33.7 -122.7 0 0 0 0 0 0 0", file=fp)
    print("0 1 LHAND_JNT_OPEN", file=fp)
    print("0 2 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(RealX, RealY), file=fp)
    #次の行でハンドの回転を指定（なぜか分からないが、関節6個なのに引数は7個にしないとズレる）
    #angleは弧度法から度数法へ変換！
    print("0 2 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + NxAngle), file=fp)
    print("0 4 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(RealZ - 0.3), file=fp)
    print("0 1 LHAND_JNT_CLOSE", file=fp)
#高く持ち上げるバージョン
    print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.45 - RealZ), file=fp)
    print("0 2 LARM_XYZ_ABS 0.5 0.20 0.45 -180 -90 145", file=fp)
    print("0 2 LARM_XYZ_REL 0 0 -0.20 0 0 0", file=fp)
    print("0 1 LHAND_JNT_OPEN", file=fp)
    print("0 2 LARM_XYZ_REL 0 0 0.20 0 0 0", file=fp)
# 低く持ち上げるバージョン
#    print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.3 - RealZ),file=fp)
#    print("0 2 LARM_XYZ_ABS 0.5 0.20 0.3 -180 -90 145",file=fp)
#    print("0 2 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
#    print("0 1 LHAND_JNT_OPEN",file=fp)
#    print("0 2 LARM_XYZ_REL 0 0 0.10 0 0 0",file=fp)
    print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 0 -33.7 -122.7 0 0 0 0 0 0 0", file=fp)
    fp.close()

def decide_grasp_position(x, y, obj_orientation, pcd_matrix_index_image, pcd_matrix):
    index = int(pcd_matrix_index_image[y][x])
    camera_loc = pcd_matrix[index] / 1000
    calib_path = data_folder_path / "cfg" / "calibmat.txt"
    grasp_position = list(transform_camera_to_robot(camera_loc, str(calib_path)))

    if obj_orientation+33 > 145+90:
        obj_orientation -= 180
    elif obj_orientation+33 < 145-90:
        obj_orientation += 180 
    yaw = round(obj_orientation, 1)

    grasp_position.append(yaw)
    print("Grasp position is ", grasp_position)

    return grasp_position

def transform_camera_to_robot(camera_loc, calib_path):
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

#各種設定=============================================
#INPUT_HEIGHT = 638  #入力深度画像の縦
#INPUT_WIDTH = 588   #入力深度画像の横
OUTPUT_HEIGHT = 125  #出力深度画像（CNNへの入力）の縦
OUTPUT_WIDTH = 125   #出力深度画像（CNNへの入力）の横
#2Dキャリブレーションなら内側底面を切り取るようにマージンを設定
#3Dキャリブレーションなら外側上面を切り取るようにマージンを設定
LEFT_MARGIN = 10#321    #箱の縁などが入力画像に写ってしまう場合，左をどれだけ無視するか
RIGHT_MARGIN = 10#1280-910  #箱の縁などが入力画像に写ってしまう場合，右をどれだけ無視するか
TOP_MARGIN = 10#263     #箱の縁などが入力画像に写ってしまう場合，上をどれだけ無視するか
BOTTOM_MARGIN = 10#1024-886  #箱の縁などが入力画像に写ってしまう場合，下をどれだけ無視するか
BETANURI_BLACK = 15 #トリミング後の画像（箱外枠含む）に、BETANURI_BLACKピクセル幅の黒枠を描画することで枠を消す
# Method0の設定=
DismissAreaWidth_Graspability = 100 # Graspabilityの探索から除外する（上下左右からの）エリア幅
# Method1の設定
WINDOW_HEIGHT = 250 #ラスタ走査するウィンドウの縦
WINDOW_WIDTH = 250  #ラスタ走査するウィンドウの横
WINDOW_STRIDE = 50  #ラスタ走査するウィンドウのずらし幅
DissmissPixelValue = 100 #画像の中央がこれ以下の輝度値ならスキップ
#全体に関わる設定
BEFORE_TO_AFTER = 500 / 225  #( シミュレーションでのdepth画像のサイズ ) / (　シミュレーション内での箱のサイズ　)
HAND_WIDTH = 60 #ハンドの開き幅 //シミュレーションと同様に40mm　ただし、Graspability計算時のハンドの形状(HAND_THICKNESS)は要調整
TOP_NUMBER = 5
CATCH_DEPTH_WINDOW = 20 #実際の距離値の取得は把持中心位置近傍の深度画像値の最大値から算出する
#カメラ関連===========================================================
#次の二行は　OniSampleUtilities.hの定義ではmm こっちではm で記述することに注意！！
MAX_DISTANCE = 960*0.001 #WindowsPCのmain.cpp（PLY→デプス画像）の定義と同じになっているか要確認 ※ただし単位をmm→Mに変換
MIN_DISTANCE = 900*0.001 #WindowsPCのmain.cpp（PLY→デプス画像）の定義と同じになっているか要確認 ※ただし単位をmm→Mに変換
StandardPointX = 0.63 #キャリブレーション時の最終コーナー（南東）のxy座標
StandardPointY = 0.10 #キャリブレーション時の最終コーナー（南東）のxy座標
BoxX = 0.23 #使用する箱の長さ（世界座標X方向）
BoxY = 0.22 #使用する箱の長さ（世界座標Y方向）

tdatetime = dt.now()
tstr = tdatetime.strftime('RESULT%Y-%m-%d_%H_%M_%S')
current_folder_path = pathlib.Path(__file__).resolve().parent
data_folder_path = (current_folder_path / ".." / "data_folder").resolve()
tstr = tdatetime.strftime("%Y-%m-%d-%H-%M-%S")
save_folder_path = data_folder_path / "result" / "Experiment" / tstr


# main
if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Learning Grasp_position and Graspability from depth image')
    #parser.add_argument('--gpu', '-gpu', default=-1, type=int,help='GPU ID (negative value indicates CPU)')  #GPU:0 CPU:-1
    #args = parser.parse_args()
    #if args.gpu >= 0:
        #cuda.check_cuda_available()
    #xp = cuda.cupy if args.gpu >= 0 else np
    xp = np
    METHOD = 0

    #端末からの引数を取得
    argv = sys.argv
    argc = len(argv)
    if argc != 2:
        print("Error : there are many argument.")
    #１つ目の引数（arg[0])はプログラム名なので無視
    #２つ目の引数はMethod番号（0→M0, 1→M1, 2→M3 を実行）
    METHOD = int(argv[1])

    tdatetime = dt.now()
    tstr = tdatetime.strftime('RESULT%Y-%m-%d_%H_%M_%S')
    current_folder_path = pathlib.Path(__file__).resolve().parent
    data_folder_path = (current_folder_path / ".." / "data_folder").resolve()
    tstr = tdatetime.strftime("%Y-%m-%d-%H-%M-%S")
    save_folder_path = data_folder_path / "result" / "Experiment" / tstr
    os.makedirs(save_folder_path)
    os.makedirs(save_folder_path/"QUARTER_DEPTH")
    os.makedirs(save_folder_path/"QUARTER_HAND")
    os.makedirs(save_folder_path/"QUARTER_HAND_WITH_WORK".format(METHOD, tstr))
    os.makedirs(save_folder_path/"HAND_WITH_WORK".format(METHOD, tstr))
    os.makedirs(save_folder_path/"tmp".format(METHOD, tstr))


    current_folder_path = pathlib.Path(__file__).resolve().parent
    data_folder_path = (current_folder_path / ".." / "data_folder").resolve()
    ply_filepath = data_folder_path / "ply" / "2022-9-18" / "1.ply"
    pcd = o3d.io.read_point_cloud(str(ply_filepath))
    pc = np.asanyarray(pcd.points)
    im, matrix_image, pcd_matrix, MaxDepth = prePly2depth.main(pc)

    # im = cv2.imread('./DEPTH_TMP/depthimg_ycam3d.png', 0)
    #im = cv2.imread('./DEPTH_TMP/MedianFilterDepth_Rotated.png',0)
    #画像サイズを取得し、設定したマージンから、トリミング後のサイズを計算
    INPUT_HEIGHT, INPUT_WIDTH = im.shape
    TRIMMED_HEIGHT = INPUT_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN
    TRIMMED_WIDTH = INPUT_WIDTH - LEFT_MARGIN - RIGHT_MARGIN

    #輪郭が箱の縁になるようにトリミング
    im_cut = im[TOP_MARGIN:INPUT_HEIGHT-BOTTOM_MARGIN, LEFT_MARGIN:INPUT_WIDTH-RIGHT_MARGIN]
    #さらに画像サイズはそのままで、輪郭を黒塗りにする（上下左右 数ピクセルを黒色化※一番最後の数字が黒色にしたいピクセル幅)
    cv2.rectangle(im_cut, (0, 0), (TRIMMED_WIDTH, TRIMMED_HEIGHT), (0, 0, 0), BETANURI_BLACK)

    black = np.zeros((im_cut.shape[0], im_cut.shape[1]))
    cv2.imwrite(str(save_folder_path/"black1280_1024.png"), black)
    #トリミング後と同じサイズの黒一色画像を用意(Method0,2実行時の最終的な把持位置を成否CNNで予測する際に使用)
    black = cv2.imread(str(save_folder_path/'black1280_1024.png'), 0)
    black_cut = black[TOP_MARGIN:INPUT_HEIGHT-BOTTOM_MARGIN, LEFT_MARGIN:INPUT_WIDTH-RIGHT_MARGIN]

    #成功予測された把持位置を記録するテキスト
    fp = open(save_folder_path/'best{}.txt'.format(TOP_NUMBER), 'wt')
    fp.close()
    cv2.imwrite(str(save_folder_path/'depth.png'), im_cut)
    img_pass = str(save_folder_path/'depth.png')
    cv2.imwrite(str(save_folder_path/'black.png'), black_cut)
    black_pass = str(save_folder_path/'black.png')
    cv2.imwrite(str(save_folder_path/'best{}.png'.format(TOP_NUMBER)), im_cut)
    best_pass = str(save_folder_path/'best{}.png'.format(TOP_NUMBER))

    #探索時間の計測開始1
    StartTime = time.time()

    #Graspabilityを計算
    Graspability(METHOD, tstr, img_pass)

    #探索時間の計測終了1
    DurationTime = time.time() - StartTime
    print("Graspabilityの探索時間は{}secでした。".format(DurationTime))

    #Graspabilityの高い把持位置をカラーで描画
    draw_best_hand_with_work(best_pass, 0, 255, 255)

    for top_count in range(TOP_NUMBER):
        #とりあえずログを記録
        fp = open(save_folder_path/'best{}.txt'.format(TOP_NUMBER), 'a')
        print("G{}位".format(top_count+1), file=fp)
        print("Graspability={:.3f} / Gマップ=G{:.0f}_{:.0f} / (x,y)=({:.0f},{:.0f}) / 回転角={:.2f}rad / 深度(0-255)={:.0f}".format(Top_Graspability[top_count]/255, Top_CountA[top_count], Top_CountB[top_count], Top_ImgX[top_count], Top_ImgY[top_count], Top_Angle[top_count], Top_ImgZ[top_count]), file=fp)
        #print("Gマップ=G{}_{} / (x,y)=({}, {}) / 回転角={}rad / 深度（0〜255）={}".format(Top_CountA[top_count],Top_CountB[top_count],Top_ImgX[top_count],Top_ImgY[top_count],Top_Angle[top_count],Top_ImgZ[top_count]),file=fp)
        print("====================================================================================================================================", file=fp)
        fp.close()

    # #最終的に確定した把持位置の「QUARTER_DEPTH画像」「QUARTER_HAND画像」「QUARTER_HAND_WITH_WORK画像」をコピー
    # final_quarter_depth = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_DEPTH/img0.png".format(METHOD, tstr), 0)
    # cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_depth.png'.format(METHOD, tstr), final_quarter_depth)
    # final_quarter_hand = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_HAND/img0.png".format(METHOD, tstr), 0)
    # cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_hand.png'.format(METHOD, tstr), final_quarter_hand)
    # final_quarter_hand_with_work = cv2.imread("./Method{}/EXPERIMENT/{}/QUARTER_HAND_WITH_WORK/img0.png".format(METHOD, tstr), 0)
    # cv2.imwrite('./Method{}/EXPERIMENT/{}/final_quarter_hand_with_work.png'.format(METHOD, tstr), final_quarter_hand_with_work)

    #モーションファイルを生成
    final_top_count = 0
    final_center_x = int(Top_ImgX[final_top_count])
    final_center_y = int(Top_ImgY[final_top_count])
    final_center_z = Top_ImgZ[final_top_count]
    final_angle = Top_Angle[final_top_count]
    grasp_position = decide_grasp_position(final_center_x, final_center_y, final_angle, matrix_image, pcd_matrix)
    # generate_motion(final_center_x, final_center_y, final_center_z, final_angle)

    #最終的な実行箇所を実験者側からわかりやすくするために回転（手法に関係なく）
    img = Image.open('{}'.format(best_pass))
    img = img.rotate(180)
    img.save(save_folder_path/'実験者目線.png')