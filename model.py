import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
	"""docstring for Generator"""
	def __init__(self, args):
		super(Generator, self).__init__()
		self.args = args
		self.g1 = nn.ConvTranspose3d(self.args.z_size, 512, kernel_size=4, stride=1)
		self.g1_bn = nn.Batchnorm3d(512)
		self.g2 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2)
		self.g2_bn = nn.Batchnorm3d(256)
		self.g3 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2)
		self.g3_bn = nn.Batchnorm3d(128)
		self.g4 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2)
		self.g4_bn = nn.Batchnorm3d(128)
		self.g5 = nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2)

	def forward(self, x):
		x = x.view(-1, self.args.z_size, 1, 1, 1)
		x = F.relu(self.g1_bn(self.g1(x)))
		x = F.relu(self.g2_bn(self.g2(x)))
		x = F.relu(self.g3_bn(self.g3(x)))
		x = F.relu(self.g4_bn(self.g4(x)))
		x = F.sigmoid(self.g5(x))
		return x 





class Discriminator(nn.Module):
	"""docstring for Discriminator"""
	def __init__(self, args):
		super(Discriminator, self).__init__()
		self.args = args
		self.cube_len = args.cube_len
		self.d1 = nn.Conv3d(1, 64, kernel_size=4, stride=2)
		self.d1_bn = nn.Batchnorm3d(64)
		self.d2 = nn.Conv3d(64, 128, kernel_size=4, stride=2)
		self.d2_bn = nn.Batchnorm3d(128)
		self.d3 = nn.Conv3d(128, 256, kernel_size=4, stride=2)
		self.d3_bn = nn.Batchnorm3d(256)
		self.d4 = nn.Conv3d(256, 512, kernel_size=4, stride=2)
		self.d4_bn = nn.Batchnorm3d(512)
		self.d5 = nn.Conv3d(512, 1, kernel_size=4, stride=1)

	def forward(self, x):
		x = x.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)
		x = F.leaky_relu(self.g1_bn(self.g1(x)), negative_slope=self.args.leak_value)
		x = F.leaky_relu(self.g2_bn(self.g2(x)), negative_slope=self.args.leak_value)
		x = F.leaky_relu(self.g3_bn(self.g3(x)), negative_slope=self.args.leak_value)
		x = F.leaky_relu(self.g4_bn(self.g4(x)), negative_slope=self.args.leak_value)
		x = F.sigmoid(self.g5(x))
		return x 





		