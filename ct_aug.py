from data_aug import *

data_nature = os.path.join(data_root, r'medicine\COVID\CT_COVID')
os.chdir(data_nature)
img = cv2.imread('2020.01.24.919183-p27-134.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img.shape)


aug_show(img, albu.ElasticTransform(alpha=1, sigma=50, alpha_affine=50,border_mode = cv2.BORDER_CONSTANT), 'elastic_transform')
aug_show(img, albu.GridDistortion(p=1, border_mode = cv2.BORDER_CONSTANT), 'grid_distortion')
