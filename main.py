from trainer import train
import argparse

def main(args): 
	train(args)


if __name__ = '__main__':
	arg_lists = []
	parser = argparse.ArgumentParser()

	def add_argument_group(name):
		arg = parser.add_argument_group(name)
		arg_lists.append(arg)
		return arg

	net_arg = add_argument_group('Network')
	net_arg.add_argument('--epochs', type=float, default=2000,
						help='number of epochs to train')
	net_arg.add_argument('--batch_size', type=float, default=100,
						help='size of the mini batch to be used')
	net_arg.add_argument('--lr_gen', type=float, default=0.0025,
						help='generator network learning rate')
	net_arg.add_argument('--lr_disc', type=float, default=0.001,
						help='discriminator network learning rate')
	net_arg.add_argument('--betas', type=float, default=(0.5,0.5),
						help='value of betas in adam optimizer')
	net_arg.add_argument('--z_size', type=float, default=200,
					help='size of input vector')
	net_arg.add_argument('--z_dis', type=str, default="norm", choices=["norm", "uni"],
						help='distribution to be used for z')
	net_arg.add_argument('--bias', type=str2bool, default=False,
						help='true: to use bias in the model')
	net_arg.add_argument('--leak_value', type=float, default=0.2,
						help='leak value in leaky relu')
	net_arg.add_argument('--cube_len', type=float, default=64,
						help='cube length of voxels')
	net_arg.add_argument('--d_thresh', type=float, default=0.8,
						help='discriminator accuracy threshold')
	net_arg.add_argument('--obj', type=str, default="chair",
						help='training dataset object category')
	net_arg.add_argument('--noisy_label', type=str2bool, default=True,
						help='true: use noisy_label')
	net_arg.add_argument('--sched', type=str2bool, default=True,
						help='true : learning rate scheduler')

	misc_arg = add_argument_group('Directories, etc')
	misc_arg.add_argument('--input_dir', type=str, default='./input',
						help='input path')
	misc_arg.add_argument('--output_dir', type=str, default="./output",
						help='output path')
	misc_arg.add_argument('--ckpt_dir', type='str', default='./ckpt',
						help='Directory in which to save model checkpoints')
	misc_arg.add_argument('--logs_dir', type=str, default='/log/',
						help='Directory in which Tensorboard logs wil be stored')
	misc_arg.add_argument('--image_dir', type=str, default='/image/',
						help='for output image path save in output_dir + image_dir')
	misc_arg.add_argument('--data_dir', type=str, default='/chair/',
						help='dataset load path')
	misc_arg.add_argument('--model_name', type=str, default="GAN",
						help='name of the model')
	misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True,
						help='true: to use tensorboard logging')
	misc_arg.add_argument('--save_freq', type=int, default=100,
						help='To save model for every n steps')
	args = parser.parse_args()
	main(args)

