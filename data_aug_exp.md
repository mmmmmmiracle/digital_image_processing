# 图像增强

***工具包：[albumentations](https://github.com/albumentations-team/albumentations#documentation)***

***[博客地址](https://www.cnblogs.com/54hys/p/12694084.html)***

<a id = 'top'></a>

## CONTENT


|                      data augmentations                      |                 link                  |                         description                          |
| :----------------------------------------------------------: | :-----------------------------------: | :----------------------------------------------------------: |
| [CenterCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.CenterCrop) |        [查看结果](#CenterCrop)        |                           中心剪裁                           |
| [Crop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Crop) |           [查看结果](#Crop)           |                         指定位置剪裁                         |
| [CropNonEmptyMaskIfExists](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.CropNonEmptyMaskIfExists) | [查看结果](#CropNonEmptyMaskIfExists) |      如果掩码为非空，则使用掩码裁剪区域，否则随机裁剪。      |
| [ElasticTransform](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ElasticTransform) |     [查看结果](#ElasticTransform)     | [Best Practices for Convolutional Neural Networks applied to Visual Document](https://dl.acm.org/doi/10.5555/938980.939477) |
| [Flip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Flip) |           [查看结果](#Flip)           |                  水平，垂直或水平和垂直翻转                  |
| [GridDistortion](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GridDistortion) |      [查看结果](#GridDistortion)      | albumentations 中主要提供了三种非刚体变换方法：ElasticTransform、GridDistortion 和 OpticalDistortion。 |
| [GridDropout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.GridDropout) |       [查看结果](#GridDropout)        |                 以网格方式删除图像的矩形区域                 |
| [HorizontalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.HorizontalFlip) |      [查看结果](#HorizontalFlip)      |                           水平翻转                           |
| [IAAAffine](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAAffine) |        [查看结果](#IAAAffine)         | 在输入上放置规则的点网格，并通过仿射变换在这些点的附近随机移动 |
| [IAACropAndPad](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAACropAndPad) |      [查看结果](#IAACropAndPad)       |                          剪裁和填充                          |
| [IAAFliplr](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAFliplr) |        [查看结果](#IAAFliplr)         |                           左右翻转                           |
| [IAAFlipud](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAFlipud) |        [查看结果](#IAAFlipud)         |                           上下翻转                           |
| [IAAPerspective](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAPerspective) |      [查看结果](#IAAPerspective)      |                  对输入执行随机四点透视变换                  |
| [IAAPiecewiseAffine](https://albumentations.readthedocs.io/en/latest/api/imgaug.html#albumentations.imgaug.transforms.IAAPiecewiseAffine) |    [查看结果](#IAAPiecewiseAffine)    | 在输入端放置一个规则的点网格，并通过仿射变换随机移动这些点的邻域 |
| [Lambda](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Lambda) |          [查看结果](#Lambda)          |                      用户自定义图像增强                      |
| [LongestMaxSize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.LongestMaxSize) |      [查看结果](#LongestMaxSize)      | 如果图像最长边小于max_size, 将最长变为max_size, 并保留长宽比resize |
| [MaskDropout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MaskDropout) |       [查看结果](#MaskDropout)        |                                                              |
| [OpticalDistortion](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.OpticalDistortion) |    [查看结果](#OpticalDistortion)     |                             畸变                             |
| [PadIfNeeded](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.PadIfNeeded) |       [查看结果](#PadIfNeeded)        |                           判断填充                           |
| [RandomCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomCrop) |        [查看结果](#RandomCrop)        |                           随机剪裁                           |
| [RandomCropNearBBox](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomCropNearBBox) |    [查看结果](#RandomCropNearBBox)    |                                                              |
| [RandomGridShuffle](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomGridShuffle) |    [查看结果](#RandomGridShuffle)     |                         网格打乱图像                         |
| [RandomResizedCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomResizedCrop) |    [查看结果](#RandomResizedCrop)     |                         剪裁并resize                         |
| [RandomRotate90](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomRotate90) |      [查看结果](#RandomRotate90)      |                         随机旋转90度                         |
| [RandomScale](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomScale) |       [查看结果](#RandomScale)        |                         随机尺度变换                         |
| [RandomSizedBBoxSafeCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSizedBBoxSafeCrop) | [查看结果](#RandomSizedBBoxSafeCrop)  |                                                              |
| [RandomSizedCrop](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomSizedCrop) |     [查看结果](#RandomSizedCrop)      |                           随机剪裁                           |
| [Resize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Resize) |          [查看结果](#Resize)          |                       重新调整图像大小                       |
| [Rotate](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Rotate) |          [查看结果](#Rotate)          |                             旋转                             |
| [ShiftScaleRotate](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ShiftScaleRotate) |     [查看结果](#ShiftScaleRotate)     |                     平移、尺度加旋转变换                     |
| [SmallestMaxSize](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.SmallestMaxSize) |     [查看结果](#SmallestMaxSize)      |               将短边变为maxsize， 并保持长宽比               |
| [Transpose](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Transpose) |        [查看结果](#Transpose)         |                             转置                             |
| [VerticalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.VerticalFlip) |       [查看结果](#VerticalFlip)       |                           垂直翻转                           |
|                                                              |                                       |                                                              |

***工具函数***

```python
import numpy as np 
import cv2
import matplotlib.pyplot as plt 
import albumentations as albu 
import os,sys

'''
	data augmentation util: albumentations
	reference: https://github.com/albumentations-team/albumentations#documentation
'''

def aug(img, aug_func):
	return aug_func(**{'image':img})['image']

def aug_compose(img, aug_func_list):
	strong_aug = albu.Compose(aug_func_list, p=1)
	return strong_aug(**{'image':img})['image']

def aug_show(img, aug_func, save_fig_name):
	plt.figure(figsize=(16,9))
	for i in range(8):
		plt.subplot(2, 4, i+1)
		img_aug = aug(img.copy(), aug_func)
		plt.imshow(img_aug)
	os.chdir(os.path.join(cur_root,'pics'))
	plt.savefig(f'{save_fig_name}.png', dpi=120)	
	plt.show()
```

***原图***

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413215946499-1984814931.png)


<a id = 'CenterCrop'></a>


## 1. CenterCrop

```python
def center_crop(img):
	height, width = img.shape[:2]
	plt.figure(figsize=(16,9))
	for i in range(8):
		plt.subplot(4, 2, i+1)
		crop_height, crop_width	= np.random.randint(100, height), np.random.randint(100, width)
		print(crop_height, crop_width)
		img_crop = aug(img.copy(), albu.CenterCrop(crop_height, crop_width,p=1))
		plt.imshow(img_crop)
	plt.show()	
	plt.savefig('center_crop.png', dpi=300)	
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413221032932-221338183.png)



[回到顶部](#top)


<a id = 'Crop'></a>

## 2. Crop

```python
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
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413223533012-1709433628.png)


[回到顶部](#top)


<a id = 'CropNonEmptyMaskIfExists'></a>

## 3. CropNonEmptyMaskIfExists

```python

```

[回到顶部](#top)

<a id = 'ElasticTransform'></a>

## 4. ElasticTransform

- alpha、sigma：高斯过滤参数，float类型
- alpha_affine：范围为 (-alpha_affine, alpha_affine)，float 类型
- interpolation、border_mode、value、mask_value：与其他类含义一样
- approximate：是否应平滑具有固定大小核的替换映射（displacement map），若启用此选项，在大图上会有两倍的速度提升，boolean类型。
- p：使用此转换的概率，默认值为 0.5

> (1) 首先需要对图像中的每个像素点(x,y)产生两个-1~1之间的随机数，Δx(x,y)和Δy(x,y)，分别表示该像素点的x方向和y方向的移动距离；
> (2) 生成一个以0为均值，以σ为标准差的高斯核k_nn，并用前面的随机数与之做卷积，并将结果作用于原图像
> 一般来说，alpha越小，sigma越大，产生的偏差越小，和原图越接近。
> [参考链接](https://blog.csdn.net/lhanchao/article/details/54234490)

```python
aug_show(img, albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50), 'elastic_transform')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413224942692-1463734522.png)


[回到顶部](#top)

<a id = 'Flip'></a>

## 5. Flip

```python
aug_show(img, albu.Flip(p=0.5), 'flip')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413231440642-972196069.png)


[回到顶部](#top)

<a id = 'GridDistortion'></a>

## 6. GridDistortion

- num_steps：在每一条边上网格单元的数量，默认值为 5，int 类型
- distort_limit：如果是单值，那么会被转成 (-distort_limit, distort_limit)，默认值为 (-0.03, 0.03)，float或float数组类型
- interpolation、border_mode、value、mask_value：与其他类含义一样
- p：使用此转换的概率，默认值为 0.5

```python
aug_show(img, albu.GridDistortion(p=1, border_mode = cv2.BORDER_CONSTANT), 'grid_distortion')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413231600309-721489143.png)


[回到顶部](#top)

<a id = 'GridDropout'></a>

## 7. GridDropout

```python
aug_show(img, albu.GridDropout(), 'grid_dropout')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413231654659-128806197.png)


[回到顶部](#top)

<a id = 'HorizontalFlip'></a>

## 8. HorizontalFlip

```python
aug_show(img, albu.HorizontalFlip(), 'horizontal_flip')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413231744901-2054348943.png)


[回到顶部](#top)

<a id = 'IAAAffine'></a>

## 9. IAAAffine

```python
aug_show(img, albu.IAAAffine(p=1, scale=0.8, translate_percent=0.8, rotate=30), 'iaa_affine')	
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413231844054-1308199791.png)


[回到顶部](#top)

<a id = 'IAACropAndPad'></a>

## 10. IAACropAndPad

```python
def iaa_crop_and_pad(img):
	plt.figure(figsize=(16,9))
	for i in range(8):
		plt.subplot(2, 4, i+1)
		img_crop = aug(img.copy(), albu.IAACropAndPad(p=1, percent=	np.random.randint(1, 20)/100))
		plt.imshow(img_crop)
	os.chdir(os.path.join(cur_root,'pics'))
	plt.savefig('iaa_crop_and_pad.png', dpi=120)	
	plt.show()
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200413231943068-1347491675.png)


[回到顶部](#top)

<a id = 'IAAFliplr'></a>

## 11. IAAFliplr

```python
aug_show(img, albu.IAAFliplr(), 'iaa_fliplr')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225315822-63805377.png)


[回到顶部](#top)

<a id = 'IAAFlipud'></a>

## 12. IAAFlipud

```python
aug_show(img, albu.IAAFlipud(), 'iaa_flipud')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225333713-2087863505.png)


[回到顶部](#top)

<a id = 'IAAPerspective'></a>

## 13. IAAPerspective

```python
aug_show(img, albu.IAAPerspective(), 'iaa_perspective')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225348855-2093564587.png)


[回到顶部](#top)

<a id = 'IAAPiecewiseAffine'></a>

## 14. IAAPiecewiseAffine

```python
aug_show(img, albu.IAAPiecewiseAffine(), 'iaa_piecewise_affine')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225402089-1267545184.png)


[回到顶部](#top)

<a id = 'Lambda'></a>

## 15. Lambda

```python
def random_brightness_or_gamma(image, **kwargs):
	seed = np.random.randint(0,2)
	aug_func = albu.RandomBrightness() if seed else albu.RandomGamma()
	# print(aug_func)
	return aug_func(**{'image': image})['image']

aug_show(img, albu.Lambda(image=random_brightness), 'lambda')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225424450-1702117850.png)


[回到顶部](#top)

<a id = 'LongestMaxSize'></a>

## 16. LongestMaxSize

```python
aug_show(img, albu.LongestMaxSize(max_size=1024), 'longest_max_size')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225438271-639395340.png)


[回到顶部](#top)

<a id = 'MaskDropout'></a>

## 17. MaskDropout



[回到顶部](#top)

<a id = 'OpticalDistortion'></a>

## 18. OpticalDistortion

```python
aug_show(img, albu.OpticalDistortion(distort_limit=0.25, shift_limit=0.25, border_mode= cv2.BORDER_CONSTANT, p=1), 'optical_distortion')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225454548-2007192699.png)


[回到顶部](#top)

<a id = 'PadIfNeeded'></a>

## 19. PadIfNeeded

```python
aug_show(img, albu.PadIfNeeded(min_height=300, min_width=400, border_mode= cv2.BORDER_CONSTANT, p=1), 'pad_if_needed')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225509166-517704186.png)


[回到顶部](#top)

<a id = 'RandomCrop'></a>

## 20. RandomCrop

```python
aug_show(img, albu.RandomCrop(height=120, width=160, p=1), 'random_crop')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225520973-483307482.png)


[回到顶部](#top)

<a id = 'RandomCropNearBBox'></a>

## 21. RandomCropNearBBox



[回到顶部](#top)

<a id = 'RandomGridShuffle'></a>

## 22. RandomGridShuffle

```python
aug_show(img, albu.RandomGridShuffle(grid=(4,4), p=1), 'random_grid_shuffle')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225534975-1351973931.png)


[回到顶部](#top)

<a id = 'RandomResizedCrop'></a>

## 23. RandomResizedCrop

```python
aug_show(img, albu.RandomResizedCrop(height=240, width=320, scale=(0.3, 1.0), ratio=(0.75, 1.25)), 'random_resized_crop')
```

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200414225545782-1947374985.png)


[回到顶部](#top)

<a id = 'RandomRotate90'></a>

## 24. RandomRotate90

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416215701789-499527306.png)


[回到顶部](#top)

<a id = 'RandomScale'></a>

## 25. RandomScale

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416215715669-557154797.png)


[回到顶部](#top)

<a id = 'RandomSizedBBoxSafeCrop'></a>

## 26. RandomSizedBBoxSafeCrop 	



[回到顶部](#top)

<a id = 'RandomSizedCrop'></a>

## 27. RandomSizedCrop

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416215943216-1186381862.png)


[回到顶部](#top)

<a id = 'Resize'></a>

## 28. Resize

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416215956779-1078538832.png)


[回到顶部](#top)

<a id = 'Resize'></a>

## 29. Resize

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416215959774-619791549.png)


[回到顶部](#top)

<a id = 'Rotate'></a>

## 30. Rotate

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416220018708-1520262111.png)

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200417113050429-489063548.png)



[回到顶部](#top)

<a id = 'ShiftScaleRotate'></a>

## 31. ShiftScaleRotate

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416220032768-1420412175.png)


[回到顶部](#top)


<a id = 'SmallestMaxSize'></a>


## 32. SmallestMaxSize

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416220419171-848292055.png)


[回到顶部](#top)


<a id = 'Transpose'></a>


## 33. Transpose

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416220133144-93917449.png)


[回到顶部](#top)


<a id = 'VerticalFlip'></a>


## 34. VerticalFlip

![](https://img2020.cnblogs.com/blog/1564250/202004/1564250-20200416220148873-1382587554.png)

[回到顶部](#top)



## References

[1] A. Buslaev, V. I. Iglovikov, E. Khvedchenya, A. Parinov, M. Druzhinin, A. A. Kalinin, Albumentations:
Fast and flexible image augmentations, Information 11 (2020). URL: https:
//www.mdpi.com/2078-2489/11/2/125. doi:10.3390/info11020125.