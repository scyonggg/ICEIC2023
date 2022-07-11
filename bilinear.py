import math
import torch
import torch.nn.functional as F

def rotate_equi(input_image, rotate_angle):
    batch = input_image.size(0)
    width = input_image.size(3)
    height = input_image.size(2)
    pi = torch.acos(torch.zeros(1)).item() * 2
    input_image = input_image.cuda()
    def yaw_offset(base, rotate_angle):
        height = base.size(2)
        width = base.size(3)
        ###### spherical coordinate #####
        
        spherical_base = torch.zeros_like(base)
        fov_y = 90

        spherical_base[:,0,:,:] = ((base[:,0,:,:]/width) * 359 - 359/2 + 180) * pi / 180 # phi
        spherical_base[:,1,:,:] = ((base[:,1,:,:] / height) * fov_y - fov_y/2 + 90 ) * pi/ 180 # theta

        rotate_angle = -180
        spherical_base[:,0,:,:] = spherical_base[:,0,:,:] + rotate_angle * pi / 180
        spherical_base[:,0,:,:] = torch.where(2 * pi < spherical_base[:,0,:,:], spherical_base[:,0, :, :] - 2 * pi, spherical_base[:,0, :, :])
        spherical_base[:,0,:,:] = torch.where(0 > spherical_base[:,0,:,:], spherical_base[:,0, :, :] + 2 * pi, spherical_base[:,0, :, :])
 
        
        
        
        spherical_base[:,0,:,:] = spherical_base[:,0,:,:] * width / (pi * 2)
        spherical_base[:,1,:,:] = (spherical_base[:,1,:,:] * 180/pi + fov_y/2 -90) * height / fov_y
        


        return spherical_base[:,0:2,:,:]


    x_base = torch.linspace(0, width, width).repeat( height, 1).cuda().unsqueeze(0).unsqueeze(0).float().repeat(batch,1,1,1)
    y_base = torch.linspace(0, height, height).repeat(width, 1).transpose(0, 1).cuda().unsqueeze(0).unsqueeze(0).float().repeat(batch,1,1,1)

    base = torch.cat((x_base,y_base),dim = 1)

    base_shift = yaw_offset(base,rotate_angle = rotate_angle)
    
    base_shift[:,0,:,:] = base_shift[:,0,:,:] / width
    base_shift[:,1,:,:] = base_shift[:,1,:,:] / height
    base_shift = base_shift.permute(0,2,3,1)

    output = F.grid_sample(input_image, 2 * base_shift - 1  , mode='bilinear',
                               padding_mode='zeros')
    
    return output

def bilinear_self_equi(input_image, depth, move_ratio, depth_sample = False):
    batch = input_image.size(0)
    width = input_image.size(3)
    height = input_image.size(2)
    pi = torch.acos(torch.zeros(1)).item() * 2

    def read_depth(disp,disp_rescale=3.):
        return depth

    def offset(base, move_ratio):

        height = base.size(2)
        width = base.size(3)
        ###### spherical coordinate #####
        
        spherical_base = torch.zeros_like(base)
        spherical_base_shift = torch.zeros_like(base)
        fov_y = 179

        spherical_base[:,2,:,:] = base[:,2,:,:] # rho = abs(depth)
        spherical_base[:,0,:,:] = ((base[:,0,:,:]/width) * 359 - 359/2 + 180) * pi / 180 # phi
        spherical_base[:,1,:,:] = ((base[:,1,:,:] / height) * fov_y - fov_y/2 + 90 ) * pi/ 180 # theta

        spherical_base_shift = spherical_base.clone()
        move_ratio_x = move_ratio[0]

############ According to the video data, change the options ###########       
        move_ratio_y = 0
#        move_ratio_y = move_ratio[1]
        move_ratio_z = 0
#        move_ratio_z = move_ratio[2]
#######################################################################
        spherical_base_shift[:,0,:,:] = torch.atan2(torch.sin(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_y / spherical_base[:,2,:,:], torch.cos(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_x / spherical_base[:,2,:,:])
        spherical_base_shift[:,0,:,:] = torch.where(0> spherical_base_shift[:,0,:,:], spherical_base_shift[:,0, :, :] + 2 * pi, spherical_base_shift[:,0, :, :])


        theta_p = torch.atan2(torch.sin(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_y / spherical_base[:,2,:,:], torch.cos(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_x / spherical_base[:,2,:,:])
        theta_p = torch.where(0> theta_p, theta_p + 2 * pi, theta_p)

         
        spherical_base_shift[:,1,:,:] = torch.atan2(torch.sin(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_y/spherical_base[:,2,:,:] ,(torch.cos(spherical_base[:,1,:,:]) - move_ratio_z/spherical_base[:,2,:,:]) * torch.sin(theta_p))
        spherical_base_shift[:,1,:,:] = torch.where(spherical_base_shift[:,1, :, :] < 0, spherical_base_shift[:,1, :, :] + pi, spherical_base_shift[:,1, :, :])

        phi_p = torch.atan2(torch.sin(spherical_base[:,0,:,:]) * torch.sin(spherical_base[:,1,:,:]) - move_ratio_y/spherical_base[:,2,:,:] ,(torch.cos(spherical_base[:,1,:,:]) - move_ratio_z/spherical_base[:,2,:,:]) * torch.sin(theta_p))
 
        phi_p = torch.where(phi_p < 0, phi_p + pi, phi_p)


        depth_ratio =  (torch.cos(spherical_base[:,1,:,:]) - move_ratio_z) / torch.cos(phi_p)
        # spherical 2 cartesian
        spherical_base_shift[:,0,:,:] = spherical_base_shift[:,0,:,:] * width / (pi * 2)
        spherical_base_shift[:,1,:,:] = (spherical_base_shift[:,1,:,:] * 180/pi + fov_y/2 -90) * height / fov_y
        

        return spherical_base_shift[:,0:2,:,:], depth_ratio


    x_base = torch.linspace(0, width, width).repeat( height, 1).cuda().unsqueeze(0).unsqueeze(0).float().repeat(batch,1,1,1)
    y_base = torch.linspace(0, height, height).repeat(width, 1).transpose(0, 1).cuda().unsqueeze(0).unsqueeze(0).float().repeat(batch,1,1,1)
    depth = read_depth(depth)

    base = torch.cat((x_base,y_base,depth),dim = 1)

    base_shift, depth_ratio = offset(base,move_ratio)
    
    base_shift[:,0,:,:] = base_shift[:,0,:,:] / width
    base_shift[:,1,:,:] = base_shift[:,1,:,:] / height
    base_shift = base_shift.permute(0,2,3,1)

    if depth_sample == True:
        input_image = depth_ratio.unsqueeze(1) * input_image

    output = F.grid_sample(input_image, 2 * base_shift - 1  , mode='bilinear',
                               padding_mode='zeros')
    
    return output, depth_ratio

