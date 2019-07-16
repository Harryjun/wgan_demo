import os,sys
from PIL import Image
import scipy.misc
from glob import glob
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#import cv2 as cv
from tensorflow.examples.tutorials.mnist import input_data

prefix = './Datas/'
def get_img(img_path, crop_h, resize_h):
	img=scipy.misc.imread(img_path).astype(np.float)
	# crop resize
	crop_w = crop_h
	#resize_h = 64
	resize_w = resize_h
	h, w = img.shape[:2]
	j = int(round((h - crop_h)/2.))
	i = int(round((w - crop_w)/2.))
	cropped_image = scipy.misc.imresize(img[j:j+crop_h, i:i+crop_w],[resize_h, resize_w])

	return np.array(cropped_image)/255.0

class face3D():
	def __init__(self):
		datapath = '/ssd/fengyao/pose/pose/images'
		self.z_dim = 100
		self.c_dim = 2
		self.size = 64
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*.jpg'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-1:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, 256, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
		#fig = self.data2fig(batch_imgs[:16,:,:])
		#plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
		#plt.close(fig)
		
		return batch_imgs

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig

class celebA():
	def __init__(self):
		datapath = prefix + 'celebA'
		self.z_dim = 100
		self.size = 64
		self.channel = 3
		self.data = glob(os.path.join(datapath, '*.jpg'))

		self.batch_count = 0

	def __call__(self,batch_size):
		batch_number = len(self.data)/batch_size
		if self.batch_count < batch_number-1:
			self.batch_count += 1
		else:
			self.batch_count = 0

		path_list = self.data[self.batch_count*batch_size:(self.batch_count+1)*batch_size]

		batch = [get_img(img_path, 128, self.size) for img_path in path_list]
		batch_imgs = np.array(batch).astype(np.float32)
		'''
		print self.batch_count
		fig = self.data2fig(batch_imgs[:16,:,:])
		plt.savefig('out_face/{}.png'.format(str(self.batch_count).zfill(3)), bbox_inches='tight')
		plt.close(fig)
		'''
		return batch_imgs

	def data2fig(self, samples):
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample)
		return fig

class mnist():
	def __init__(self, flag='conv', is_tanh = False):
		self.datapath = prefix + 'data_pic/'
		self.X_dim = 784 # for mlp
		self.z_dim = 100
		self.y_dim = 10
		self.sizex = 96 # for conv
		self.sizey = 128 # for conv
		self.channel = 3 # for conv
		#self.data = input_data.read_data_sets(datapath, one_hot=True)
		self.flag = flag
		self.is_tanh = is_tanh
		self.Train_nums = 50
		self.train_data = np.zeros([self.Train_nums,self.sizex,  self.sizey,self.channel]) 
		img_list = os.listdir(self.datapath)
		count = 0
		for i in range(self.Train_nums):
			img_path = os.path.join(self.datapath, img_list[i])  # 图片文件
			#CV图片处理方式

			#img = cv.imread(img_path)
			#self.train_data[count, :, :,:]  = cv.resize(img,(self.sizey,  self.sizex))
			
			#Python图片处理

			img = Image.open(img_path)
			self.train_data[count, :, :,:] = img.resize((self.sizey,self.sizex))
			#gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)			
			print(count)
			count+=1

	def __call__(self,batch_size):
		batch_imgs = self.next_batch(self.datapath,batch_size)
		#batch_imgs,y = self.next_batch(prefix,batch_size)
		if self.flag == 'conv':
			batch_imgs = np.reshape(batch_imgs, (batch_size, self.sizex, self.sizey, self.channel)) 
		if self.is_tanh:
			batch_imgs = batch_imgs*2 - 1
		#return batch_imgs, y
		return batch_imgs

	def next_batch(self,data_path, batch_size):
	#def next_batch(self,data_path, lable_path, batch_size):
		train_temp = np.random.randint(low=0, high=self.Train_nums, size=batch_size) # 生成元素的值在[low,high)区间，随机选取
		train_data_batch = np.zeros([batch_size,self.sizex,  self.sizey,self.channel]) # 其中[img_row,  img_col, 3]是原数据的shape，相应变化
		#train_label_batch = np.zeros([batch_size, self.size, self.size]) #
		count = 0 # 后面就是读入图像，并打包成四维的batch
		#print(data_path)
		img_list = os.listdir(data_path)
		#print(img_list)

		# 图片提前存储到内存中
		for i in train_temp:
			train_data_batch[count, :, :,:]  = self.train_data[i]
			count+=1
		return train_data_batch#, train_label_batch

		#通用版代码
		'''		
		for i in train_temp:
			img_path = os.path.join(data_path, img_list[i])  # 图片文件
			img = cv.imread(img_path)
			gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
			train_data_batch[count, :, :]  = cv.resize(gray,(self.sizey,  self.sizex))
			count+=1
		return train_data_batch#, train_label_batch
		'''
	def data2fig(self, samples):
		if self.is_tanh:
			samples = (samples + 1)/2
		fig = plt.figure(figsize=(4, 4))
		gs = gridspec.GridSpec(4, 4)
		gs.update(wspace=0.05, hspace=0.05)

		for i, sample in enumerate(samples):
			ax = plt.subplot(gs[i])
			plt.axis('off')
			ax.set_xticklabels([])
			ax.set_yticklabels([])
			ax.set_aspect('equal')
			plt.imshow(sample.reshape(self.sizex,self.sizey,self.channel), cmap='Greys_r')
		return fig	

if __name__ == '__main__':
	data = face3D()
	print(data(17).shape)
