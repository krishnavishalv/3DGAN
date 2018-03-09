import os
from tqdm import trange

import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from model import Generator, Discriminator
from utils import generateZ, ShapeNetDataset, var_or_cuda

class MultiStepLR(_LRScheduler):
    """Set the learning rate of each parameter group to the initial lr decayed
    by gamma once the number of epoch reaches one of the milestones.
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        milestones (list): List of epoch indices. Must be increasing.
        gamma (float): Multiplicative factor of learning rate decay.
    Example:
        >>> # Assuming optimizer uses lr = 0.5 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 80
        >>> # lr = 0.0005   if epoch >= 80
        >>> scheduler = MultiStepLR(optimizer, milestones=[30,80], gamma=0.1)
        >>> for epoch in range(100):
        >>>     scheduler.step()
        >>>     train(...)
        >>>     validate(...)
    """

    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * self.gamma ** bisect_right(self.milestones, self.last_epoch)
                for base_lr in self.base_lrs]

def train(args):
	if args.use_tensorboard:
		import tensorflow as tf

		summary_writer = tf.summary.FileWriter(args.output_dir + args.log_dir + log_param)

		def inject_summary(summary_writer, tag, value, step):
				summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
				summary_writer.add_summary(summary, global_step=step)

		inject_summary = inject_summary


	data_path = args.input_dir + args.data_dir + "train/"
	dataset = ShapeNetDataset(data_path, args)
	shape_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

	Gen = Generator(args)
	Disc = Discriminator(args)
	print('[*] Number of parameters in Generator: {:,}'.format(
			sum([p.data.nelement() for p in Gen.parameters()])))
	print('[*] Number of parameters in Discriminator: {:,}'.format(
			sum([p.data.nelement() for p in Disc.parameters()])))

	criterion = nn.BCELoss()
	
	Gen_optim = optim.Adam(Gen.parameters(), lr=args.lr_gen, betas=args.betas)
	Disc_optim = optim.Adam(Disc.parameters(), lr=args.lr_disc, betas=args.betas)

	if args.sched:
		Disc_scheduler = optim.MultiStepLR(Disc_optim, milestones=[500,1000,1500,2000])

	Gen.cuda()
	Disc.cuda()
	criterion.cuda()

	for epoch in trange(args.epochs):
		for i, x in enumerate(shape_loader):

			real_labels = torch.ones(args.batch_size).cuda()
			fake_labels = torch.ones(args.batch_size).cuda()

			if args.noisy_label: 
				real_labels = torch.Tensor(args.batch_size).uniform_(0.7, 1.2)
				fake_labels = torch.Tensor(args.batch_size).uniform_(0, 0.3)

			x = x.cuda()
			z = generateZ(args)

			Disc_real = Disc(x)
			Disc_real_loss = criterion(Disc_real, real_labels)

			fake = Gen(z)
			Disc_fake = Disc(fake)
			Disc_fake_loss = criterion(Disc_fake, fake_labels)

			Disc_loss_total = Disc_real_loss + Disc_fake_loss

			real_acc = torch.ge(Disc_real.squeeze(), 0.5).float()
			fake_acc = torch.ge(Disc_fake.squeeze(), 0.5).float()
			total_acc = torch.mean(torch.cat((real_acc, fake_acc),0))

			if total_acc <= args.d_thresh:
				Disc.zero_grad()
				Disc_loss_total.backward()
				Disc_optim.step()



			z = generateZ(args)

			fake = Gen(z)
			Disc_fake = Disc(fake)
			Gen_loss = criterion(Disc_fake, real_labels)

			Gen.zero_grad()
			Disc.zero_grad()
			Gen_loss.backward()
			Gen_optim.step()


		iteration = str(Gen_optim.state_dict())['state'][Gen_optim.state_dict()['param_groups'][0]['params'][0]['step']]
		print('Iter-{}; , Dis_loss : {:.4}, Gen_loss : {:.4}, Dis_acc : {:.4}, Disc_lr : {:.4}'.format(iteration, Disc_loss_total.data[0], Gen_loss.data[0], total_acc.data[0], Disc_optim.state_dict()['param_groups'][0]["lr"]))
		save_checkpoint()

		
def save_checkpoint(args, state):
	print("[*] Saving model to {}".format(args.ckpt_dir))
	if not os.path.exists(args.ckpt_dir):
		os.makedirs(path)
	file_name = args.model_name + '_model_best.pth.tar'
	ckpt_path = os.path.join(args.ckpt_dir, file_name)
	torch.save(state, ckpt_path)







