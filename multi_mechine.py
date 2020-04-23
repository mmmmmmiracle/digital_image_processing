import os, sys

from data_aug import *

def detect_and_descript(img, method = 'sift'):
	kp, des = None, None

	if method == 'sift':
		sift = cv2.xfeatures2d.SIFT_create()
		# kp 关键点， des 描述子
		kp, des = sift.detectAndCompute(img, None)
	elif method == 'surf':
		surf = cv2.xfeatures2d.SURF_create()
		# kp 关键点， des 描述子
		kp, des = surf.detectAndCompute(img, None)
	elif method == 'orb':
		orb = cv2.ORB_create()
		# kp 关键点， des 描述子
		kp, des = orb.detectAndCompute(img, None)
	elif method == 'brief':
		# 初始化STAR检测器
		# detector = cv2.xfeatures2d.StarDetector_create()
		# detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
		detector = cv2.FastFeatureDetector_create()

		# 初始化BRIEF特征提取器
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		# 使用STAR寻找特征点
		kp = detector.detect(img,None)

		# kp 关键点， des 描述子
		kp, des = brief.compute(img, kp)
	else:
		pass

	return kp, des 

def matching(img, kp, des, matcher, ratio):
	(img_ori, img_aug), (kp_ori,kp_aug), (des_ori, des_aug) = img, kp, des 
	try:
		matches = matcher.knnMatch(des_ori, des_aug, k=2)
		# print(type(matches), len(matches[0]), matches[0])
		good_points = []
		for m, n in matches:
			if m.distance < ratio * n.distance:
				good_points.append([m])

		img_kp_map_matches = cv2.drawMatchesKnn(img_ori.copy(), kp_ori, img_aug.copy(), kp_aug, matches, None, flags=2)
		img_kp_map_good_points = cv2.drawMatchesKnn(img_ori.copy(), kp_ori, img_aug.copy(), kp_aug, good_points, None, flags=2)
		return img_kp_map_matches, img_kp_map_good_points, good_points, True
	except Exception as e:
		return None, None, None, False

def image_align(img, kp, good_points):
	img_ori, img_aug = img
	kp_ori, kp_aug = kp 
	flag = True
	if(len(good_points) > 4):
		try:
			pts_ori = np.float32([kp_ori[m[0].queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
			pts_aug = np.float32([kp_aug[m[0].trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
			ransacReprojThreshold = 2 # 表示一对内群点所能容忍的最大投影误差
			H, status =cv2.findHomography(pts_ori,pts_aug,cv2.RANSAC,ransacReprojThreshold);
			img_align = cv2.warpPerspective(img_aug, H, (img_ori.shape[1],img_ori.shape[0]),flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
		except Exception as e:
			flag = False
	else:
		flag = False
	if flag:
		return img_align,H,status, True
	return None, None, None, False

bf = cv2.BFMatcher()

FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params,search_params)

if __name__ == "__main__":
	data_multi_machine = os.path.join(data_root, r'多机数据')
	os.chdir(data_multi_machine)
	img_name = 'out_window'
	img_oppo = cv2.imread(f'{img_name}_oppo.jpg')
	img_onep = cv2.imread(f'{img_name}_oneplus.jpg')
	img_oppo_gray = cv2.cvtColor(img_oppo, cv2.COLOR_BGR2GRAY)
	img_onep_gray = cv2.cvtColor(img_onep, cv2.COLOR_BGR2GRAY)

	method = 'sift'

	kp_oppo, des_oppo = detect_and_descript(img_oppo_gray, method=method)
	kp_onep, des_onep = detect_and_descript(img_onep_gray, method=method)

	img_kp_map_matches, img_kp_map_good_points, good_points, ret = matching((img_oppo, img_onep), (kp_oppo, kp_onep), (des_oppo, des_onep), 
							matcher = bf, ratio = 0.6)
	if(ret):
		img_align, H, status, ret = image_align((img_oppo, img_onep), (kp_oppo, kp_onep), good_points)
		if ret:
			img_match = np.vstack((img_kp_map_matches, img_kp_map_good_points))
			_img_oppo, _img_align = np.zeros_like(img_onep), np.zeros_like(img_onep)
			_img_oppo[:img_oppo.shape[0], :img_oppo.shape[1]] = img_oppo
			_img_align[:img_oppo.shape[0], :img_oppo.shape[1]] = img_align
			print(_img_oppo.shape, _img_align.shape)
			cv2.imwrite(f"{img_name}_{method}_match.png", img_match)
			cv2.imwrite(f"{img_name}_{method}_align.png", np.hstack((_img_oppo, img_onep, _img_align)))

