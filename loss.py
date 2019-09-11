import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torchvision.models as models 

vgg = models.vgg16(pretrained=True)
# print(vgg)
new_model = nn.Sequential(*list(vgg.features[:34]))#删掉网络的最后一层
new_model = new_model.cuda()
for param in new_model.parameters(): 
    param.requires_grad = False 
# print(new_model)






# pretrained_dict = vgg.state_dict()  
# model_dict = model.state_dict()  
# pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}  
# model_dict.update(pretrained_dict)  
# model.load_state_dict(model_dict)
# print(model)



class MonodepthLoss(nn.modules.Module):
    def __init__(self, n=4, SSIM_w=0.98, disp_gradient_w=1.0, lr_w=1.0):
        super(MonodepthLoss, self).__init__()
        self.SSIM_w = SSIM_w
        self.disp_gradient_w = disp_gradient_w
        self.lr_w = lr_w
        self.n = n
        self.normalize = nn.BatchNorm2d(2)
        self.sigmoid = torch.nn.Sigmoid()

    def scale_pyramid(self, img, num_scales):
        scaled_imgs = [img]#left-image as target
        for i in range(num_scales - 1):
            scaled_imgs.append(img)
        return scaled_imgs

    def gradient_x(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 1, 0, 0), mode="replicate")
        gx = img[:,:,:,:-1] - img[:,:,:,1:]  # NCHW
        return gx

    def gradient_y(self, img):
        # Pad input to keep output size consistent
        img = F.pad(img, (0, 0, 0, 1), mode="replicate")
        gy = img[:,:,:-1,:] - img[:,:,1:,:]  # NCHW
        return gy
    # def gradient_x2(self, img):
    #     # Pad input to keep output size consistent
    #     img = F.pad(img, (0, 1, 0, 0), mode="replicate")
    #     gx = img[:,:,:,:-2]+img[:,:,:,2:] - 2*img[:,:,:,1:-1]  # NCHW
    #     return gx  

    # def gradient_y2(self, img):
    #     # Pad input to keep output size consistent
    #     img = F.pad(img, (0, 0, 0, 1), mode="replicate")
    #     gy = img[:,:,:-2,:]+img[:,:,2:,:] - 2*img[:,:,1:-1,:]  # NCHW
        # return gy 
    def apply_disparity(self, img, disp):
        batch_size, _, height, width = img.size()
        #print(img.shape)
        #print(disp.shape)
        # Original coordinates of pixels
        x_base = torch.linspace(0, 1, width).repeat(batch_size, height, 1).type_as(img)
        #print(x_base)
        y_base = torch.linspace(0, 1, height).repeat(batch_size, width, 1).transpose(1, 2).type_as(img)
        #print(y_base)
        # Apply shift in X direction
        #x_shifts = disp[:, 0, :, :]
        #print(disp)
        x_shifts = disp[:, 0, :, :]  # Disparity is passed in NCHW format with 1 channel
        #print(x_shifts)
        flow_field = torch.stack((x_base + x_shifts, y_base), dim=3)
        #flow_field = torch.stack((x_base, y_base), dim=3)
        #print(flow_field.shape)
        #print(flow_field)
        # In grid_sample coordinates are assumed to be between -1 and 1
        output = F.grid_sample(img, 2*flow_field - 1, mode='bilinear', padding_mode='zeros')
        #print(output.shape)
        return output

    def generate_image_left(self, img, disp):
        return self.apply_disparity(img, -disp)

    def generate_image_right(self, img, disp):
        return self.apply_disparity(img, disp)

    def SSIM(self, x, y):
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(3, 1)(x)
        mu_y = nn.AvgPool2d(3, 1)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(3, 1)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(3, 1)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(3, 1)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d

        return torch.clamp((1 - SSIM) / 2, 0, 1)

    def disp_smoothness(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]

        weights_x = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x]
        weights_y = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y]

        smoothness_x = [disp_gradients_x[i] * weights_x[i]
                        for i in range(self.n)]
        smoothness_y = [disp_gradients_y[i] * weights_y[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x[i]) + torch.abs(smoothness_y[i])
                for i in range(self.n)]

    def disp_smoothness2(self, disp, pyramid):
        disp_gradients_x = [self.gradient_x(d) for d in disp]
        disp_gradients_y = [self.gradient_y(d) for d in disp]
        disp_gradients_x2 = [self.gradient_x(d) for d in disp_gradients_x]
        disp_gradients_y2 = [self.gradient_y(d) for d in disp_gradients_y]

        image_gradients_x = [self.gradient_x(img) for img in pyramid]
        image_gradients_y = [self.gradient_y(img) for img in pyramid]
        image_gradients_x2 = [self.gradient_x(img) for img in image_gradients_x]
        image_gradients_y2 = [self.gradient_y(img) for img in image_gradients_y]

        weights_x2 = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_x2]
        weights_y2 = [torch.exp(-torch.mean(torch.abs(g), 1,
                     keepdim=True)) for g in image_gradients_y2]

        smoothness_x2 = [disp_gradients_x2[i] * weights_x2[i]
                        for i in range(self.n)]
        smoothness_y2 = [disp_gradients_y2[i] * weights_y2[i]
                        for i in range(self.n)]

        return [torch.abs(smoothness_x2[i]) + torch.abs(smoothness_y2[i])
                for i in range(self.n)]
    def disp_smoothness3(self, disp, pyramid):
     
        disp_gradients_x3 = [self.gradient_x(d) for d in disp]
        disp_gradients_y3 = [self.gradient_y(d) for d in disp]

        image_gradients_x3 = [self.gradient_x(img) for img in pyramid]
        image_gradients_y3 = [self.gradient_y(img) for img in pyramid]




        hhhx=[disp_gradients_x3[i]-image_gradients_x3[i] for i in range(self.n)]
        hhhy=[disp_gradients_y3[i]-image_gradients_y3[i]  for i in range(self.n)]
        

        return [torch.abs(hhhx[i]) + torch.abs(hhhy[i])
                for i in range(self.n)]                        

    def forward(self, input, target,epoch):
        """
        Args:
            input [disp1, disp2, disp3, disp4]
            target [left, right]

        Return:
            (float): The loss
        """
        #kk= [input]
        kk=input

        left,right = target  #1*3*256*512
 
      
   


        left_pyramid = self.scale_pyramid(left, self.n)
   
        right_pyramid = self.scale_pyramid(right, self.n)

        # Prepare disparities
        
        disp_left_est = [d[:,0,:,:].unsqueeze(1)/d.size()[3] for d in kk]#disp change to 4 images

        disp_right_est = [d[:,1,:,:].unsqueeze(1)/d.size()[3] for d in kk]
     
        self.disp_left_est = disp_left_est
        self.disp_right_est = disp_right_est
      
        # Generate images
        # plt.imshow(flipleft[0].squeeze().cpu().detach().numpy())
        # plt.show()
        # plt.imshow(disp_left_est[0].squeeze().cpu().detach().numpy())
        # plt.show()
        
        left_est = [self.generate_image_left(right_pyramid[i],
                    disp_left_est[i]) for i in range(self.n)]
        preloss_left=[new_model(left_est[i][:,:,:,160:-160])  for i in range(self.n)]
        preloss1_left=[new_model(left_pyramid[i][:,:,:,160:-160])  for i in range(self.n)]
        # plt.imshow(np.transpose(preloss1_left[0].squeeze().cpu().detach().numpy(),(1,2,0)))
        # plt.show()
        # plt.imshow(np.transpose(preloss_left[0].squeeze().cpu().detach().numpy(),(1,2,0)))
        # plt.show()
        leftpre=[torch.mean((preloss_left[i]-preloss1_left[i])*(preloss_left[i]-preloss1_left[i]))/ 1.42 ** i   for i in range(self.n)]
        # wqxleft=[ (torch.mean(left_est[i],1,keepdim=True)!=0).float()            for i in range(self.n)]  
        # # for i in range(self.n):
        # #     wqxleft[i][:,:,:,100:]=1
        # plt.imshow(wqxleft[0].squeeze().cpu().detach().numpy())
        # plt.show()   
       
        right_est = [self.generate_image_right(left_pyramid[i],
                     disp_right_est[i]) for i in range(self.n)]
        preloss_right=[new_model(right_est[i][:,:,:,160:-160]) for i in range(self.n)]
        preloss1_right=[new_model(right_pyramid[i][:,:,:,160:-160]) for i in range(self.n)]
        rightpre=[torch.mean((preloss_right[i]-preloss1_right[i])*(preloss_right[i]-preloss1_right[i]))/ 1.42 ** i   for i in range(self.n)]
        preloss=sum(leftpre+rightpre)
        wqxright=[ torch.mean(right_est[i][:,:,:,160:-160]/ 1.42 ** i )          for i in range(self.n)]    
        wqxleft=[ torch.mean(left_est[i][:,:,:,160:-160]/ 1.42 ** i )          for i in range(self.n)] 
        dloss=sum(wqxright+wqxleft)
        # for i in range(self.n):
        #         wqxright[i][:,:,:,:-100]=1
      
        self.left_est = left_est
        self.right_est = right_est
        # L1
        #*wqxleft[i]
        #*wqxright[i]
        l1_left = [torch.abs(left_est[i] - left_pyramid[i]) / 1.42 ** i
                   for i in range(self.n)]
        l1_right = [torch.abs(right_est[i] - right_pyramid[i]) / 1.42 ** i
                    for i in range(self.n)]
        l1_left1 = [ torch.mean(l1_left[i][:,:,:,160:-160] )        for i in range(self.n)]
        l1_right1 = [ torch.mean(l1_right[i][:,:,:,160:-160])     for i in range(self.n)]            
        #gradient

        gradientleftwqx=self.disp_smoothness3(left_est,
                     left_pyramid)
        gradientrightwqx=self.disp_smoothness3(right_est,
                     right_pyramid)
        #*wqxleft[i]
        #*wqxright[i]
        gradientleftwqx1 = [torch.mean(torch.mean(torch.abs(
                          gradientleftwqx[i][:,:,:,160:-160]),1,keepdim=True) )/ (1.42 ** i)
                          for i in range(self.n)]
                    
        gradientrightwqx1 = [torch.mean(torch.mean(torch.abs(
                          gradientrightwqx[i][:,:,:,160:-160]),1,keepdim=True)) / 1.42** i
                           for i in range(self.n)]             
        
        # SSIM

       
        #*wqxleft[i][:,:,1:-1,1:-1]
        #*wqxright[i][:,:,1:-1,1:-1]
        ssim_left = [torch.mean(self.SSIM(left_est[i][:,:,:,160:-160],
                     left_pyramid[i][:,:,:,160:-160]))/ 1.42 ** i for i in range(self.n)]
  
        ssim_right = [torch.mean(self.SSIM(right_est[i][:,:,:,160:-160],
                      right_pyramid[i][:,:,:,160:-160]))/ 1.42 ** i for i in range(self.n)]
        image_loss_left = [self.SSIM_w * ssim_left[i]+0.15*l1_left1[i]
                           +0.15*gradientleftwqx1[i]
                           for i in range(self.n)]
        image_loss_right = [self.SSIM_w * ssim_right[i]+0.15*l1_right1[i]
                            +0.15*gradientrightwqx1[i]
                            for i in range(self.n)]              
   

      

        image_loss = sum(image_loss_left + image_loss_right)
       
        
        
        # L-R Consistency

        
        left_disp = [self.generate_image_left(right_est[i],
                           disp_left_est[i]) for i in range(self.n)]
        # plt.imshow(np.transpose(left_disp[0].squeeze().cpu().detach().numpy(),(1,2,0)))
        # plt.show()                     
        right_disp = [self.generate_image_right(left_est[i],
                           disp_right_est[i]) for i in range(self.n)]
        # L-R Consistency
        #*wqxleft[i]
        #*wqxright[i]
        lr_left_loss = [torch.mean(torch.mean(torch.abs(left_disp[i][:,:,:,160:-160]
                        - left_pyramid[i][:,:,:,160:-160]),1,keepdim=True))/1.42 ** i for i in range(self.n)]
        lr_right_loss = [torch.mean(torch.mean(torch.abs(right_disp[i][:,:,:,160:-160]
                         - right_pyramid[i][:,:,:,160:-160]),1,keepdim=True))/1.42 ** i for i in range(self.n)]
        lr_loss = sum(lr_left_loss + lr_right_loss)
        

        # Disparities smoothness
        disp_left_smoothness = self.disp_smoothness2(disp_left_est,
                                                    left_pyramid)
        disp_right_smoothness = self.disp_smoothness2(disp_right_est,
                                                     right_pyramid)
                                            
        # Disparities smoothness
        disp_left_loss = [torch.mean(torch.abs(
                          disp_left_smoothness[i][:,:,:,160:-160])) / 1.42 ** i
                          for i in range(self.n)]
                 
        disp_right_loss = [torch.mean(torch.abs(
                          disp_right_smoothness[i][:,:,:,160:-160])) / 1.42** i
                           for i in range(self.n)]
        disp_gradient_loss = sum(disp_left_loss + disp_right_loss)
        if epoch >=0 and epoch <=30 :
            aaaaa=0.001
        if epoch>30 and epoch<=100:
            aaaaa=0.1
        if epoch >100 and epoch <=115:
            aaaaa=0  
        if epoch >200 and epoch <=215:
            aaaaa=0 
        if epoch >300 and epoch <=315:
            aaaaa=0 
        if epoch >400 and epoch <=415:
              aaaaa=0 
        if epoch >500 and epoch <=515:
             aaaaa=0 
        if epoch >600 and epoch <=615:
              aaaaa=0 
        if epoch >700 and epoch <=715:
              aaaaa=0 
        if epoch >800 and epoch <=815:
             aaaaa=0 
        if epoch >900 and epoch <=915:
                aaaaa=0      
        
        else:
            aaaaa=0.1   
             
        loss = 1*image_loss +  1.5 * lr_loss+aaaaa*disp_gradient_loss+0.001*dloss+0.3*preloss
        
        self.image_loss = image_loss
        self.disp_gradient_loss = disp_gradient_loss
        self.lr_loss = lr_loss

        return loss
