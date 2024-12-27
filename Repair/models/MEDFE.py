
import torch

from collections import OrderedDict

from .base_model import BaseModel
from . import networks
from .loss import VGG16, PerceptualLoss, StyleLoss, GANLoss, PerceptualLoss_2


class MEDFE(BaseModel):
    def __init__(self, opt):
        super(MEDFE, self).__init__(opt)
        self.isTrain = opt.isTrain
        self.opt = opt
        self.device = torch.device('cuda')
        # define tensors
        self.vgg = VGG16()
        self.PerceptualLoss = PerceptualLoss()
        self.PerceptualLoss_2 = PerceptualLoss_2()
        self.StyleLoss = StyleLoss()
        self.input_DE = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_ST = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.Gt_Local = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.Gt_DE = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.Gt_ST = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.mask_global = self.Tensor(opt.batchSize, 1, opt.fineSize, opt.fineSize)
        self.model_names = []

        if len(opt.gpu_ids) > 0:
            self.use_gpu = True
            self.vgg = self.vgg.to(self.gpu_ids[0])
            self.vgg = torch.nn.DataParallel(self.vgg, self.gpu_ids)
        # load/define networks  EN:Encoder DE:Decoder  MEDFE: Mutual Encoder Decoder with Feature Equalizations
        self.netEN, self.netDE, self.netMEDFE, self.stde_loss = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.norm,
                                                                                  opt.use_dropout, opt.init_type,
                                                                                  self.gpu_ids,
                                                                                  opt.init_gain,opt.batchSize)
        #裁剪80x80,
        self.netmaskEN = networks.define_G2(opt.input_mask_nc,opt.ngf,opt.norm,opt.init_type,self.gpu_ids,opt.init_gain)

        #sideoutput上采用模块
        self.sideoutput1 =  networks.side_up1(opt.in_channel1,3)
        self.sideoutput2 =  networks.side_up2(opt.in_channel2,3)


        self.model_names=['EN', 'DE', 'MEDFE','maskEN']


        if self.isTrain:

            self.netD = networks.define_D_1(3, opt.ndf,
                                          opt.n_layers_D, opt.norm, opt.init_type, self.gpu_ids, opt.init_gain)
            self.netF = networks.define_patch_D(3, opt.ndf,
                                          opt.n_layers_D, opt.norm, opt.init_type, self.gpu_ids, opt.init_gain)
            self.model_names.append('D')
            self.model_names.append('F')
        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = GANLoss(tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            self.optimizer_EN = torch.optim.Adam(self.netEN.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_DE = torch.optim.Adam(self.netDE.parameters(),
                                                 lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_MEDFE = torch.optim.Adam(self.netMEDFE.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
            #新加主干的优化器 maskEN
            self.optimizer_maskEN  = torch.optim.Adam(self.netmaskEN.parameters(),
                                                      lr=opt.lr,betas=(opt.beta1,0.999))

            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_EN)
            self.optimizers.append(self.optimizer_DE)
            self.optimizers.append(self.optimizer_MEDFE)

            self.optimizers.append(self.optimizer_maskEN)

            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_F)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            # print('---------- Networks initialized -------------')
            # networks.print_network(self.netEN)
            # networks.print_network(self.netDE)
            # networks.print_network(self.netMEDFE)
            # if self.isTrain:
            #     networks.print_network(self.netD)
            #     networks.print_network(self.netF)
            # print('-----------------------------------------------')
            #####modified
        if self.isTrain:
            if opt.continue_train :
                print('Loading pre-trained network!')
                self.load_networks(self.netEN, 'EN', opt.which_epoch)
                self.load_networks(self.netDE, 'DE', opt.which_epoch)
                self.load_networks(self.netMEDFE, 'MEDFE', opt.which_epoch)
                self.load_networks(self.netD, 'D', opt.which_epoch)
                self.load_networks(self.netF, 'F', opt.which_epoch)
                self.load_networks(self.netmaskEN,"maskEN",opt.which_epoch)

    def name(self):
        return self.modlename

    def mask_process(self, mask):
        mask = mask[:,0:1,:,:]
        # mask = torch.unsqueeze(mask, 0)
        # mask = torch.unsqueeze(mask, 1)
        mask = mask.byte()
        return mask


    def set_input(self, input_De, input_St, mask,x,y,epoch,FLY_img):    #De是原图，St是遮挡图

        self.Gt_DE = input_De.to(self.device)
        self.Gt_ST = input_St.to(self.device)
        self.input_DE = input_De.to(self.device)
        self.mask1= mask.to(self.device)
        self.fly_img = FLY_img.to(self.device)
        self.mask_global = self.mask_process(mask.to(self.device))
        self.Gt_Local = input_De.to(self.device)
        self.G_mask   = input_St.to(self.device)

        self.epoch = epoch
        # define local area which send to the local discriminator
        self.crop_x = x
        self.crop_y = y

        # if 58 < self.crop_x < 148 and 58 < self.crop_y < 148:
        #     self.crop_x = self.crop_x-(self.crop_x-58)
        #     self.crop_y = self.crop_y-(self.crop_y-58)
        #8个邻域的顶点
        self.crop_x1 = x-32
        self.crop_y1 = y-32
        self.crop_x2 = x
        self.crop_y2 = y-32
        self.crop_x3 = x+32
        self.crop_y3 = y-32

        self.crop_x4 = x-32
        self.crop_y4 = y
        self.crop_x5 = x+32
        self.crop_y5 = y

        self.crop_x6 = x-32
        self.crop_y6 = y+32
        self.crop_x7 = x
        self.crop_y7 = y+32
        self.crop_x8 = x+32
        self.crop_y8 = y+32

        #裁剪出八个邻域
        self.mask_list = []
        self.mask_1 = self.G_mask[:,:,self.crop_x1:self.crop_x1+32,self.crop_y1:self.crop_y1+32]
        self.mask_2 = self.G_mask[:, :, self.crop_x2:self.crop_x2 + 32, self.crop_y2:self.crop_y2 + 32]
        self.mask_3 = self.G_mask[:, :, self.crop_x3:self.crop_x3 + 32, self.crop_y3:self.crop_y3 + 32]
        self.mask_4 = self.G_mask[:, :, self.crop_x4:self.crop_x4 + 32, self.crop_y4:self.crop_y4 + 32]
        self.mask_5 = self.G_mask[:, :, self.crop_x5:self.crop_x5 + 32, self.crop_y5:self.crop_y5 + 32]
        self.mask_6 = self.G_mask[:, :, self.crop_x6:self.crop_x6 + 32, self.crop_y6:self.crop_y6 + 32]
        self.mask_7 = self.G_mask[:, :, self.crop_x7:self.crop_x7 + 32, self.crop_y7:self.crop_y7 + 32]
        self.mask_8 = self.G_mask[:, :, self.crop_x8:self.crop_x8 + 32, self.crop_y8:self.crop_y8 + 32]
        self.mask_list.append(self.mask_1)
        self.mask_list.append(self.mask_2)
        self.mask_list.append(self.mask_3)
        self.mask_list.append(self.mask_4)
        self.mask_list.append(self.mask_5)
        self.mask_list.append(self.mask_6)
        self.mask_list.append(self.mask_7)
        self.mask_list.append(self.mask_8)




        self.ex_mask = self.mask_global.expand(self.mask_global.size(0), 3, self.mask_global.size(2),
                                               self.mask_global.size(3))
        #  unpositve with original mask
        self.inv_ex_mask = torch.add(torch.neg(self.ex_mask.float()), 1).float()
        # set loss groundtruth for two branch
        self.stde_loss[0].set_target(self.Gt_DE, self.Gt_ST)
        # Do not set the mask regions as 0
        self.input_DE.narrow(1, 0, 1).masked_fill_(self.mask_global.narrow(1, 0, 1).bool(), 2 * 123.0 / 255.0 - 1.0)
        self.input_DE.narrow(1, 1, 1).masked_fill_(self.mask_global.narrow(1, 0, 1).bool(), 2 * 104.0 / 255.0 - 1.0)
        self.input_DE.narrow(1, 2, 1).masked_fill_(self.mask_global.narrow(1, 0, 1).bool(), 2 * 117.0 / 255.0 - 1.0)

    def forward(self):


        fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.netEN(
            torch.cat([self.Gt_ST, self.mask1,self.fly_img], 1))

        #论文中的input是原图+mask拼接成6通道
        # fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6 = self.netEN(self.Gt_ST)

        #8个邻域拼接

        mask_cat = torch.cat([self.mask_list[1],self.mask_list[2],
                              self.mask_list[3], self.mask_list[4], self.mask_list[5],
                              self.mask_list[6], self.mask_list[7]
                              ],dim=1)

        # mask_80  = self.Gt_DE[:,:,self.crop_x-20:self.crop_x+60,self.crop_y-20:self.crop_y+60]

        out_feature = self.netmaskEN(mask_cat)

        De_in = [fake_p_1, fake_p_2, fake_p_3, fake_p_4, fake_p_5, fake_p_6]
        x_out = self.netMEDFE(De_in, self.mask_global)
        # x_out = self.netMEDFE(De_in)
        self.fake_out,out_feature_1,out_feature_2= self.netDE(x_out[0], x_out[1], x_out[2], x_out[3], x_out[4], x_out[5],out_feature)

        self.up_side1  = self.sideoutput1(out_feature_1)
        self.up_side2  = self.sideoutput2(out_feature_2)



        # #显示输入图
        # in_put = self.Gt_ST.detach().cpu().numpy().squeeze()
        # in_put = np.transpose(cv2.normalize(in_put,None,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U),(1,2,0))
        # cv2.imwrite("in_put.jpg",in_put)
        # #显示mask
        # in_mask = self.mask1.detach().cpu().numpy().squeeze()
        # in_mask = np.transpose(cv2.normalize(in_mask,None,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U),(1,2,0))
        # in_mask2 = self.mask_global.detach().cpu().numpy().squeeze()
        # in_mask2 = np.transpose(cv2.normalize(in_mask2, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U),
        #                        )
        # cv2.imwrite("int_mask1.png",in_mask)
        # cv2.imwrite("int_mask2.png", in_mask2)
        # #显示特征图



    def backward_D(self):
        fake_AB = self.fake_out
        real_AB = self.Gt_DE  # GroundTruth
        # #显示一下裁剪的部位
        # out_show = real_local.detach().cpu().numpy().squeeze()
        # out_show =np.transpose(cv2.normalize(out_show,None,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U),(1,2,0))
        # cv2.imwrite("out.jpg",out_show)
        # #显示一下裁剪的部位

        # #显示一下输入的图片
        # out_show = self.Gt_ST.detach().cpu().numpy().squeeze()
        # out_show = np.transpose(cv2.normalize(out_show,None,0,255,norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_8U),(1,2,0))
        # cv2.imwrite("out.jpg",out_show)

        # Global Discriminator
        self.pred_fake = self.netF(fake_AB.detach())
        self.pred_real = self.netF(real_AB)
        self.loss_D_fake = self.criterionGAN(self.pred_fake, self.pred_real, True)

        #裁剪出mask原图和fake的mask图
        real_mask = self.Gt_DE[:,:,self.crop_x:self.crop_x+32,self.crop_y:self.crop_y+32]
        fake_mask = self.fake_out[:,:,self.crop_x:self.crop_x+32,self.crop_y:self.crop_y+32]
        #计算gan_loss
        self.real_mask_f = self.netD(real_mask)
        self.fake_mask_f = self.netD(fake_mask.detach())
        self.real_mask_loss = self.criterionGAN(self.real_mask_f,self.fake_mask_f,True)


        self.loss_D = self.loss_D_fake + self.real_mask_loss

        self.loss_D.backward()

    def backward_G(self):
        # First, The generator should fake the discriminator
        # real_AB = self.Gt_DE
        # fake_AB = self.fake_out
        #
        # real_local = self.Gt_Local
        # fake_local = self.fake_out[:, :, self.crop_x:self.crop_x + 40, self.crop_y:self.crop_y + 40]

        # Global discriminator
        # pred_real = self.netD(real_AB)
        # pred_fake = self.netD(fake_AB)
        # Local discriminator
        # pred_real_F = self.netF(real_local)
        # pred_fake_f = self.netF(fake_local)

        # self.loss_G_GAN = self.criterionGAN(pred_fake, pred_real, False) + self.criterionGAN(pred_fake_f, pred_real_F,
        #                                                                                      False)    self.loss_G_GAN * self.opt.lambda_Gan
        # Second, Reconstruction loss

        #mask位置做L1_loss
        mask_local = self.Gt_DE[:,:,self.crop_x:self.crop_x+32,self.crop_y:self.crop_y+32]
        mask_fake  = self.fake_out[:,:,self.crop_x:self.crop_x+32,self.crop_y:self.crop_y+32]

        #sideoupt和原图做L1loss
        self.sideoutputL1 = self.criterionL1(self.up_side1,self.Gt_DE)
        self.sideoutputL2 = self.criterionL1(self.up_side2,self.Gt_DE)

        self.mask_L1 = self.criterionL1(mask_fake,mask_local)
        self.loss_L1 = self.criterionL1(self.fake_out, self.Gt_DE)
        self.Perceptual_loss = self.PerceptualLoss(self.fake_out, self.Gt_DE)

        self.Perceptual_loss_mask = self.PerceptualLoss_2(mask_fake,mask_local)

        # self.Style_Loss = self.StyleLoss(self.fake_out, self.Gt_DE)

        # self.loss_G = self.loss_G_L1 + self.loss_G_GAN *0.2 + self.Perceptual_loss * 0.2 + self.Style_Loss *250

        self.loss_G_L1 = self.mask_L1 + self.loss_L1/64 + self.sideoutputL2/64 + self.sideoutputL1/64
        self.loss_G_P = self.Perceptual_loss * self.opt.lambda_P*0.1 +self.Perceptual_loss_mask*0.1

        self.stde_loss_value = 0
        for loss in self.stde_loss:
            self.stde_loss_value += loss.backward()
            self.stde_loss_value += loss.loss
        self.loss_G_L1 += self.stde_loss_value

        # #先有L1_loss
        # if self.epoch <= 5:
        #     self.loss_G = self.loss_G_L1*0.1
        #     self.loss_G.backward()
        # else:
        #     self.loss_G = self.loss_G_L1*0.1 + self.loss_G_P
        #     self.loss_G.backward()
        self.loss_G = self.loss_G_P + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        # Optimize the D and F first
        self.set_requires_grad(self.netF, True)
        self.set_requires_grad(self.netD, True)
        self.set_requires_grad(self.netEN, False)
        self.set_requires_grad(self.netDE, False)
        self.set_requires_grad(self.netMEDFE, False)


        self.optimizer_D.zero_grad()
        self.optimizer_F.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.optimizer_F.step()

        # Optimize  EN, DE, MEDEF
        self.set_requires_grad(self.netF, False)
        self.set_requires_grad(self.netD, False)
        self.set_requires_grad(self.netEN, True)
        self.set_requires_grad(self.netDE, True)
        self.set_requires_grad(self.netMEDFE, True)
        self.set_requires_grad(self.netmaskEN,True)
        self.optimizer_EN.zero_grad()
        self.optimizer_DE.zero_grad()
        self.optimizer_MEDFE.zero_grad()
        self.optimizer_maskEN.zero_grad()
        self.backward_G()
        self.optimizer_maskEN.step()
        self.optimizer_MEDFE.step()
        self.optimizer_EN.step()
        self.optimizer_DE.step()

    def get_current_errors(self):
        # show the current loss
        return OrderedDict([
                            ('G_Loss', self.loss_G.data),
                            # ('G_L1_loss', self.loss_G_L1),
                            # ('G_P_loss', self.loss_G_P),
                            # # ('F', self.loss_F_fake.data),
                            ('D_Loss', self.loss_D.data)
                            ])

    # You can also see the Tensorborad
    def get_current_visuals(self):
        input_image = (self.Gt_ST.data.cpu()+1)/2.0
        fake_image = (self.fake_out.data.cpu()+1)/2.0
        real_gt = (self.Gt_DE.data.cpu()+1)/2.0
        mask_show = (self.mask1.cpu()+1)/2.0
        fly_img  = (self.fly_img.cpu()+1)/2.0
        return input_image, fake_image, real_gt,mask_show,fly_img
    
