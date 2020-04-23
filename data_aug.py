import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import albumentations as albu 
import os,sys
import math
from math import *

'''
	data augmentation util: albumentations
	reference: https://github.com/albumentations-team/albumentations#documentation
'''

cur_root = os.path.abspath('./')
data_root = os.path.abspath(r'..\dataset')

def aug(img, aug_func):
	return aug_func(**{'image':img})['image']

def aug_compose(img, aug_func_list):
	strong_aug = albu.Compose(aug_func_list, p=1)
	return strong_aug(**{'image':img})['image']

def some_trying(img):
	img_shift = aug(img.copy(), 
					albu.ShiftScaleRotate(shift_limit = 0.5 ,rotate_limit=0, scale_limit=0.1 ,p=1, border_mode=cv2.BORDER_CONSTANT))
	img_rotate = aug(img.copy(), albu.Rotate(limit=30, p=1))
	img_resize_up =  aug(img.copy(), albu.Resize(360, 480))
	img_resize_down = aug(img.copy(), albu.Resize(180, 240))
	img_vflip = aug(img.copy(), albu.VerticalFlip(p=1))
	img_hflip = aug(img.copy(), albu.HorizontalFlip(p=1))
	img_affine = aug(img.copy(), albu.IAAAffine(p=1, scale=0.8, translate_percent=0.8, rotate=30))
	img_gridshuffle = aug(img.copy(), albu.RandomGridShuffle(p=1))
	img_griddistortion = aug(img.copy(), albu.GridDistortion(p=1, border_mode = cv2.BORDER_CONSTANT))
	img_elastictransform = aug(img.copy(), albu.ElasticTransform(p=1, alpha = 16, sigma = 2,border_mode = cv2.BORDER_CONSTANT))
	# img = aug_compose(img, [albu.Resize(360, 480), albu.VerticalFlip(p=1)])

	# img_list = [img_shift, img_rotate, img_resize_up, img_resize_down, 
	# 			img_vflip, img_hflip, img_affine, img_gridshuffle]

	img_list = ['img_shift', 'img_rotate', 'img_resize_up', 'img_resize_down', 
				'img_vflip', 'img_hflip', 'img_affine', 'img_gridshuffle',
				'img_griddistortion', 'img_elastictransform']

	plt.figure(figsize=(16, 9))
	for i, img in enumerate(img_list):
		plt.subplot(2, 5, i+1)
		plt.imshow(eval(img))
		plt.title(img)
	plt.show()

def center_crop(img):
	height, width = img.shape[:2]
	plt.figure(figsize=(16,9))
	for i in range(8):
		plt.subplot(2, 4, i+1)
		crop_height, crop_width	= np.random.randint(100, height), np.random.randint(100, width)
		print(crop_height, crop_width)
		img_crop = aug(img.copy(), albu.CenterCrop(crop_height, crop_width,p=1))
		plt.imshow(img_crop)
	os.chdir(os.path.join(cur_root,'pics'))
	plt.savefig('center_crop.png', dpi=120)	
	plt.show()	

def crop(img):
	height, width = img.shape[:2]
	plt.figure(figsize=(16,9))
	for i in range(8):
		plt.subplot(2, 4, i+1)
		x_min, y_min = np.random.randint(24, 120), np.random.randint(16, 80)
		x_max, y_max = np.random.randint(x_min, width), np.random.randint(y_min, height)	
		img_crop = aug(img.copy(), albu.Crop(x_min, y_min, x_max, y_max, p=1.0))
		plt.imshow(img_crop)
	os.chdir(os.path.join(cur_root,'pics'))
	plt.savefig('crop.png', dpi=120)	
	plt.show()

def iaa_crop_and_pad(img):
	plt.figure(figsize=(16,9))
	for i in range(8):
		plt.subplot(2, 4, i+1)
		img_crop = aug(img.copy(), albu.IAACropAndPad(p=1, percent=	np.random.randint(1, 20)/100))
		plt.imshow(img_crop)
	os.chdir(os.path.join(cur_root,'pics'))
	plt.savefig('iaa_crop_and_pad.png', dpi=120)	
	plt.show()

def aug_show(img, aug_func, save_fig_name):
	plt.figure(figsize=(16,9))
	for i in range(8):
		plt.subplot(2, 4, i+1)
		img_aug = aug(img.copy(), aug_func)
		plt.imshow(img_aug)
	os.chdir(os.path.join(cur_root,'pics'))
	plt.savefig(f'{save_fig_name}.png', dpi=120)	
	plt.show()

def random_brightness_or_gamma(image, **kwargs):
	seed = np.random.randint(0,2)
	aug_func = albu.RandomBrightness() if seed else albu.RandomGamma()
	# print(aug_func)
	return aug_func(**{'image': image})['image']

def rotate_with_all_info(angle=30, **kwargs):   
	'''旋转， 保留图像所有信息'''
	def rotate(image, angle=angle, **kwargs):
		angle = np.random.randint(-angle,angle)
		height, width = image.shape[:2]
		height_new = int(width * fabs(sin(radians(angle))) + height * fabs(cos(radians(angle))))
		width_new = int(height * fabs(sin(radians(angle))) + width * fabs(cos(radians(angle))))

		mat_rotate = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)

		# 因为旋转之后,坐标系原点是新图像的左上角,所以需要根据原图做转化
		mat_rotate[0, 2] += (width_new - width) / 2
		mat_rotate[1, 2] += (height_new - height) / 2

		img_rotate = cv2.warpAffine(image, mat_rotate, (width_new, height_new),
		                     borderValue=(0, 0, 0))
		return img_rotate
	return rotate

print(cur_root)
data_nature = os.path.join(data_root, r'nature\iccv09Data\images')
print(data_nature)
os.chdir(data_nature)
img = cv2.imread('0100851.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)

center_crop(img)

if __name__=='__main__':
	data_nature = os.path.join(data_root, r'nature\iccv09Data\images')
	print(data_nature)
	os.chdir(data_nature)
	img = cv2.imread('0100851.jpg')
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	print(img.shape)

	center_crop(img)
	# crop(img)
	# aug_show(img, albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50), 'elastic_transform')
	# aug_show(img, albu.Flip(p=0.5), 'flip')
	# aug_show(img, albu.GridDistortion(p=1, border_mode = cv2.BORDER_CONSTANT), 'grid_distortion')
	# aug_show(img, albu.GridDropout(), 'grid_dropout')
	# aug_show(img, albu.HorizontalFlip(), 'horizontal_flip')	
	# aug_show(img, albu.IAAAffine(p=1, scale=0.8, translate_percent=0.8, rotate=30), 'iaa_affine')	
	# iaa_crop_and_pad(img)
	# aug_show(img, albu.IAAFliplr(), 'iaa_fliplr')
	# aug_show(img, albu.IAAFlipud(), 'iaa_flipud')
	# aug_show(img, albu.IAAPerspective(), 'iaa_perspective')
	# aug_show(img, albu.IAAPiecewiseAffine(), 'iaa_piecewise_affine')
	# aug_show(img, albu.Lambda(image=random_brightness), 'lambda')
	# aug_show(img, albu.LongestMaxSize(max_size=1024), 'longest_max_size')
	# aug_show(img, albu.MaskDropout(), 'mask_dropout')
	# aug_show(img, albu.OpticalDistortion(distort_limit=0.25, shift_limit=0.25, border_mode= cv2.BORDER_CONSTANT, p=1), 'optical_distortion')
	# aug_show(img, albu.PadIfNeeded(min_height=300, min_width=400, border_mode= cv2.BORDER_CONSTANT, p=1), 'pad_if_needed')
	# aug_show(img, albu.RandomCrop(height=120, width=160, p=1), 'random_crop')
	# aug_show(img, albu.RandomGridShuffle(grid=(4,4), p=1), 'random_grid_shuffle')
	# aug_show(img, albu.RandomResizedCrop(height=240, width=320, scale=(0.3, 1.0), ratio=(0.75, 1.25)), 'random_resized_crop')
	# aug_show(img, albu.RandomRotate90(), 'random_rotate')
	# aug_show(img, albu.RandomScale(scale_limit = 0.5, p=1), 'random_scale')
	# aug_show(img, albu.RandomSizedCrop(min_max_height=(120, 240), height=180, width = 240, w2h_ratio=4/3), 'random_sized_crop')
	# aug_show(img, albu.Resize(480, 640), 'resize')
	# aug_show(img, albu.Rotate(border_mode=cv2.BORDER_CONSTANT, p=1), 'rotate')
	# aug_show(img, albu.ShiftScaleRotate(shift_limit = 0.5 ,rotate_limit=0, scale_limit=0.1 ,p=1, border_mode=cv2.BORDER_CONSTANT), 
	# 									'shift_scale_rotate')
	# aug_show(img, albu.SmallestMaxSize(max_size=480), 'smallest_max_size')
	# aug_show(img, albu.Transpose(), 'transpose')
	# aug_show(img, albu.VerticalFlip(), 'vertical_flip')


	# aug_show(img, albu.GaussNoise(p=1), 'gauss')
	# aug_show(img, albu.IAAAdditiveGaussianNoise(p=1), 'iaa_additive_gaussian_noise')
	# aug_show(img, albu.ISONoise(p=1), 'isonoise')

	# aug_show(img, albu.RandomFog(p=1), 'random_fog')
	# aug_show(img, albu.RandomSnow(p=1), 'random_snow')
	# aug_show(img, albu.RandomRain(rain_type='drizzle', p=1), 'random_rain')
	# aug_show(img, albu.RandomBrightness(p=1), 'random_brightness')
	# aug_show(img, albu.RandomBrightnessContrast(p=1), 'random_brightness_contrast')
	# aug_show(img, albu.RandomGamma(p=1), 'random_gamma')
	# aug_show(img, albu.RandomSunFlare(flare_roi = (0.95,0, 1, 0.05) ,p=1), 'random_sun_flare')
	# aug_show(img, albu.RandomShadow(p=1), 'random_shadow')

	# aug_show(img, albu.Lambda(image=rotate_with_all_info(angle=30)), 'rotate_with_all_info')

	img_rotate = rotate_with_all_info(30)(img.copy())
	plt.imshow(img_rotate)
	plt.axis('off')
	plt.show()



