import model 
import torch
import torch.nn.functional as F

torch.backends.cudnn.benchmark=True

net=model.U_net.U_net()

net=net.cuda()

input_tensor=torch.zeros((1,4,1600,2400)).cuda()
# output_tensor=net(input_tensor)


#four dimision padding
def padding_images(images,padding_exp=16):
    height=(images.size(2)//padding_exp+1)*padding_exp
    width=(images.size(3)//padding_exp+1)*padding_exp
    images=F.pad(images,(0,width,0,height))
    return images

def cut_images(images,padding_exp=32):
    hegith=images.size(2)-images.size(2)%padding_exp
    width=images.size(3)-images.size(3)%padding_exp
    return images[:,:,:height,:width]

#inference and get image
def inference(images,ratio):
    if(len(images.size())==3):
        images=images.unsequence(0)
    images=images*ratio
    # images=padding_images(images)
    output_images=net(images)
    # output_images=cut_images(output_images.detach())
    return output_images.detach().permute(0,3,1,2).cpu().numpy()


output=inference(input_tensor,1.0)
print(output.shape)



        
