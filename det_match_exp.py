import os, sys
import numpy as np 
import cv2	
import matplotlib.pyplot as plt 
import time, tqdm, json
# sys.path.append(os.path.abspath('../'))
from data_aug import aug, aug_compose, albu, cur_root, data_root, rotate_with_all_info
from de_fog import de_fog

def make_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)

def get_img_list(list_dir):
	return os.listdir(list_dir)

def create_mnist_list():
	res = {}
	cur_data_root = os.path.join(data_root, 'mnist')
	os.chdir(cur_data_root)
	for root1 in ['mnist_train', 'mnist_test']:
		for root2 in range(10):
			tmp_root = os.path.join((os.path.join(cur_data_root,root1)), str(root2))
			res[tmp_root] = os.listdir(tmp_root)
			print(tmp_root, len(res[tmp_root]))
	json_str = json.dumps(res)
	os.chdir(cur_root)
	with open('mnist_list.json', 'w') as json_file:
	    json_file.write(json_str)
# create_mnist_list()

def create_under_water_list():
	res = {}
	cur_data_root = os.path.join(data_root, 'nature\\under_water\\train\\image')
	res[cur_data_root] = get_img_list(cur_data_root)
	os.chdir(cur_root)
	json_str = json.dumps(res)
	with open('under_water_list.json', 'w') as json_file:
	    json_file.write(json_str)
# create_under_water_list()

def create_pcb_list():
	res = {}
	cur_data_root = os.path.join(data_root, r'engineer\PCB_DATASET\images')
	for cat in ['Missing_hole', 'Mouse_bite', 'Open_circuit', 'Short', 'Spur', 'Spurious_copper']:
		tmp_root = os.path.join(cur_data_root, cat)
		res[tmp_root] = get_img_list(tmp_root)
		print(tmp_root)
	json_str = json.dumps(res)
	os.chdir(cur_root)
	with open('pcb_list.json', 'w') as json_file:
	    json_file.write(json_str)
# create_pcb_list()

def create_covid_list():
	res = {}
	cur_data_root = os.path.join(data_root, r'medicine\COVID')
	for cat in ['CT_COVID', 'CT_NonCOVID']:
		tmp_root = os.path.join(cur_data_root, cat)
		res[tmp_root] = get_img_list(tmp_root)
		print(tmp_root)
	json_str = json.dumps(res)
	os.chdir(cur_root)
	with open('covid_list.json', 'w') as json_file:
	    json_file.write(json_str)
# create_covid_list()

def create_plant_list():
	res = {}
	cur_data_root = os.path.join(data_root, r'nature\植物病理\images')
	res[cur_data_root] = get_img_list(cur_data_root)
	json_str = json.dumps(res)
	os.chdir(cur_root)
	with open('plant_list.json', 'w') as json_file:
	    json_file.write(json_str)
# create_plant_list()

def create_heaven_list():
	res = {}
	cur_data_root = r'F:\数据集\城市区域分类\train\img'
	for cat in [str(i) for i in range(1,10)]:
		tmp_root = os.path.join(cur_data_root, f'00{cat}')
		res[tmp_root] = get_img_list(tmp_root)
		print(tmp_root)
	json_str = json.dumps(res)	
	os.chdir(cur_root)
	with open('heaven_list.json', 'w') as json_file:
	    json_file.write(json_str)
# create_heaven_list()

def create_ground_list():
	res = {}
	cur_data_root = r'F:\TIEI\数字图像处理\大作业\dataset\nature\iccv09Data\images'
	res[cur_data_root] = get_img_list(cur_data_root)
	json_str = json.dumps(res)
	os.chdir(cur_root)
	with open('ground_list.json', 'w') as json_file:
	    json_file.write(json_str)
# create_ground_list()

def create_texture_list():
	res = {}
	cur_data_root = r'F:\TIEI\数字图像处理\大作业\dataset\nature\texture'
	res[cur_data_root] = get_img_list(cur_data_root)
	json_str = json.dumps(res)
	os.chdir(cur_root)
	with open('txture_list.json', 'w') as json_file:
	    json_file.write(json_str)
# create_texture_list()

def prepare_img(img_src, resize=None):
	img_ori = cv2.imread(img_src)
	# img_aug = rotate_with_all_info(30)(img_ori)
	# img_aug = aug(img_ori, albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, border_mode=cv2.BORDER_CONSTANT))
	# img_aug = aug_compose(img_ori, [albu.ShiftScaleRotate(shift_limit = 0.2 ,rotate_limit=0, scale_limit=0 ,p=1, border_mode=cv2.BORDER_CONSTANT)]) # under_water
	# img_aug = aug_compose(img_ori, [albu.GridDistortion(p=1, border_mode = cv2.BORDER_CONSTANT)]) # pcb 
	# img_aug = aug(img_ori, albu.Lambda(image=rotate_with_all_info(30))) # plant
	# img_aug = aug(img_ori, albu.HorizontalFlip(p=1))
	# img_aug = aug(img_ori, albu.VerticalFlip(p=1)) # heaven
	# img_aug = aug(img_ori, albu.IAAAffine(p=1, scale=0.8, translate_percent=0.8, rotate=30)) # texture
	img_aug = aug_compose(img_ori, [
			albu.ShiftScaleRotate(shift_limit = 0.2 ,rotate_limit=0, scale_limit=0.2 ,p=1, border_mode=cv2.BORDER_CONSTANT),
			albu.Lambda(rotate_with_all_info(30)),
			albu.RandomFog(p=1),
			albu.RandomShadow(p=1),
			albu.GaussNoise(p=1)
		]) # mnist

	if resize is not None:
		img_ori = aug(img_ori, albu.Resize(*resize))
		img_aug = aug(img_aug, albu.Resize(*resize))

	img_ori_gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
	img_aug_gray = cv2.cvtColor(img_aug, cv2.COLOR_BGR2GRAY)

	return img_ori, img_aug, img_ori_gray, img_aug_gray


def detect_and_descript(img_src, method='sift', is_draw_kp=False):
	img_ori, img_aug, img_ori_gray, img_aug_gray = prepare_img(img_src, resize=(256,256))
	# img_ori, img_aug, img_ori_gray, img_aug_gray = prepare_img(img_src)

	kp_ori, kp_aug, des_ori, des_aug = None, None, None, None
	img_kp = None

	if method == 'sift':
		sift = cv2.xfeatures2d.SIFT_create()
		# kp 关键点， des 描述子
		kp_ori, des_ori = sift.detectAndCompute(img_ori, None)
		kp_aug, des_aug = sift.detectAndCompute(img_aug, None)
	elif method == 'surf':
		surf = cv2.xfeatures2d.SURF_create()
		# kp 关键点， des 描述子
		kp_ori, des_ori = surf.detectAndCompute(img_ori, None)
		kp_aug, des_aug = surf.detectAndCompute(img_aug, None)
	elif method == 'orb':
		orb = cv2.ORB_create()
		# kp 关键点， des 描述子
		kp_ori, des_ori = orb.detectAndCompute(img_ori, None)
		kp_aug, des_aug = orb.detectAndCompute(img_aug, None)
	elif method == 'brief':
		# 初始化STAR检测器
		# detector = cv2.xfeatures2d.StarDetector_create()
		# detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True,type=cv2.FAST_FEATURE_DETECTOR_TYPE_9_16)
		detector = cv2.FastFeatureDetector_create()

		# 初始化BRIEF特征提取器
		brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
		# 使用STAR寻找特征点
		kp_ori = detector.detect(img_ori_gray,None)
		kp_aug = detector.detect(img_aug_gray,None)

		# kp 关键点， des 描述子
		kp_ori, des_ori = brief.compute(img_ori, kp_ori)
		kp_aug, des_aug = brief.compute(img_aug, kp_aug)
	else:
		pass

	if is_draw_kp:
		img_ori_kp = cv2.drawKeypoints(img_ori.copy(), kp_ori, img_ori.copy(), color=(255,0,0))
		img_aug_kp = cv2.drawKeypoints(img_aug.copy(), kp_aug, img_aug.copy(), color=(255,0,0))
		img_kp = np.hstack((img_ori_kp, img_aug_kp))

	return (img_ori, img_aug), (kp_ori,kp_aug), (des_ori, des_aug), img_kp

def matching(img, kp, des, matching_func, ratio):
	(img_ori, img_aug), (kp_ori,kp_aug), (des_ori, des_aug) = img, kp, des 
	try:
		matches = matching_func.knnMatch(des_ori, des_aug, k=2)
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

def test(dec_method='surf', matching_func=bf, ratio=0.6):
	with open("mnist_list.json",'r', encoding='UTF-8') as f:
		res = json.load(f)
	start = time.time()
	for tmp_root, img_list in res.items():
		os.chdir(tmp_root)
		for img_src in tqdm.tqdm(img_list):
			img, kp, des, img_kp = detect_and_descript(img_src, method=dec_method, is_draw_kp=True)
			img_kp_map_matches, img_kp_map_good_points, good_points, ret = matching(img, kp, des, matching_func, ratio)
			if ret:
				img_match = np.vstack((img_kp_map_matches, img_kp_map_good_points))
				img_align, _, _, ret = image_align(img, kp, good_points)
				if ret:
					img_ori_aug_align = np.hstack((img[0], img[1], img_align))
					# cv2.imshow('img_ori_aug_align', img_ori_aug_align)
					# cv2.imshow('img_match', img_match)
					cv2.imwrite(f"{img_src.split('.')[0]}_{dec_method}_match.png", img_match)
					cv2.imwrite(f"{img_src.split('.')[0]}_{dec_method}_align.png", img_ori_aug_align)

			# cv2.imwrite()
			# print(os.getcwd())
			# print(os.path.abspath('../'))
			# cv2.imshow('img_kp', img_kp)
			# cv2.imshow('img_kp_map_matches', )
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()
			# break
		break
	print('time consumed: ', time.time() - start)

test()