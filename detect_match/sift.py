import os, sys
import numpy as np 
import cv2	
import matplotlib.pyplot as plt 
import time
sys.path.append(os.path.abspath('../'))
from data_aug import aug, albu


data_root = os.path.abspath('./')
os.chdir(os.path.join(data_root, 'input'))

img_ori = cv2.imread('ori.jpg')
img_rotate = cv2.imread('rotate.jpg')
# print(img_ori.shape, img_rotate.shape)
img_ori = aug(img_ori, albu.Resize(360, 480))
img_rotate = aug(img_rotate, albu.Resize(360, 480))
# print(img_ori.shape, img_rotate.shape)

img_ori_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
img_rotate_gray = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2GRAY)

sift = cv2.xfeatures2d.SIFT_create()
# kp 关键点， des 描述子
kp_ori, des_ori = sift.detectAndCompute(img_ori, None)
kp_rotate, des_rotete = sift.detectAndCompute(img_rotate, None)

cv2.imshow('gray_scale', np.hstack((img_ori_gray, img_rotate_gray)))
cv2.waitKey(0)
cv2.destroyAllWindows()

img_ori_kp = cv2.drawKeypoints(img_ori, kp_ori, img_ori, color=(255,0,0))
img_rotate_kp = cv2.drawKeypoints(img_rotate, kp_rotate, img_rotate, color=(255,0,0))

cv2.imshow('keypoints', np.hstack((img_ori_kp, img_rotate_kp)))
cv2.waitKey(0)
cv2.destroyAllWindows()

def matching(match_func, ratio=0.6):
	# bf_matcher = cv2.BFMatcher()
	matches = match_func.knnMatch(des_ori, des_rotete, k=2)
	# print(type(matches), len(matches[0]), matches[0])
	good_points = []
	for m, n in matches:
		if m.distance < ratio * n.distance:
			good_points.append([m])

	img_kp_map_matches = cv2.drawMatchesKnn(img_ori, kp_ori, img_rotate, kp_rotate, matches, None, flags=2)
	img_kp_map_good_points = cv2.drawMatchesKnn(img_ori, kp_ori, img_rotate, kp_rotate, good_points, None, flags=2)
	return img_kp_map_matches, img_kp_map_good_points

'''brute-force matcher'''
img_kp_map_matches, img_kp_map_good_points = matching(cv2.BFMatcher())
cv2.imshow('bf_matches', img_kp_map_matches)
cv2.imshow('bf_matches_good_points', img_kp_map_good_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''FLANN(fast library for approximate nearest neighbor) matcher'''
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)
img_kp_map_matches, img_kp_map_good_points = matching(flann)
cv2.imshow('flann_matches', img_kp_map_matches)
cv2.imshow('flann_matches_good_points', img_kp_map_good_points)
cv2.waitKey(0)
cv2.destroyAllWindows()

