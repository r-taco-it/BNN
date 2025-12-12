import torch
import pdb
import torch.nn as nn
import math
from scipy.stats import truncnorm
import numpy as np
from torch.autograd import Variable
from torch.autograd.function  import Function, InplaceFunction

import numpy as np




class Binarize(InplaceFunction):

    #def forward(ctx,input,quant_mode='det',allow_scale=False,inplace=False):
    def forward(ctx, input, quant_mode='det', allow_scale=False, inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()      

        scale= output.abs().max() if allow_scale else 1

        if quant_mode=='det':
            return output.div(scale).sign().mul(scale)
        else:
            # Stochastic Binarization using truncated Gaussian approximation (GPU-compatible)
            normalized = output.abs().max()
            normed = output.div(scale).add(1).div(2)

            # Truncated Gaussian noise approximation in PyTorch
            a, b = -6, 6  # Truncate at ±6σ
            #std_adj = (noise_std / 512) * 20 / 2  # Adjusted std similar to original logic
            #std_adj = (noise_std / 512) / 2  # Adjusted std similar to original logic
            noise_mean=0.047/576
            noise_std=5.36/576/2
            #std_adj = (noise_std)  # Adjusted std similar to 

            # Generate truncated noise (clipped normal)
            noise = torch.empty_like(output).normal_(mean=noise_mean, std=noise_std)
            noise = noise.clamp(a * noise_std + noise_mean, b * noise_std + noise_mean)

            normed = normed + noise
            binarized = normed.clamp(0, 1).round().mul(2).add(-1).mul(scale)
            return binarized

    def backward(ctx,grad_output):
        #STE 
        grad_input=grad_output
        return grad_input,None,None,None



class Quantize(InplaceFunction):
    def forward(ctx,input,quant_mode='det',numBits=4,inplace=False):
        ctx.inplace = inplace
        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()
        scale=(2**numBits-1)/(output.max()-output.min())
        output = output.mul(scale).clamp(-2**(numBits-1)+1,2**(numBits-1))
        if quant_mode=='det':
            output=output.round().div(scale)
        else:
            output=output.round().add(torch.rand(output.size()).add(-0.5)).div(scale)
        return output
    
    def backward(grad_output):
        #STE 
        grad_input=grad_output
        return grad_input,None,None

def binarized(input,quant_mode='det'):
      return Binarize.apply(input,quant_mode)  

def quantize(input,quant_mode,numBits):
      return Quantize.apply(input,quant_mode,numBits) 

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss,self).__init__()
        self.margin=1.0

    def hinge_loss(self,input,target):
            #import pdb; pdb.set_trace()
            output=self.margin-input.mul(target)
            output[output.le(0)]=0
            return output.mean()

    def forward(self, input, target):
        return self.hinge_loss(input,target)

class SqrtHingeLossFunction(Function):
    def __init__(self):
        super(SqrtHingeLossFunction,self).__init__()
        self.margin=1.0

    def forward(self, input, target):
        output=self.margin-input.mul(target)
        output[output.le(0)]=0
        self.save_for_backward(input, target)
        loss=output.mul(output).sum(0).sum(1).div(target.numel())
        return loss

    def backward(self,grad_output):
       input, target = self.saved_tensors
       output=self.margin-input.mul(target)
       output[output.le(0)]=0
       grad_output.resize_as_(input).copy_(target).mul_(-2).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(input.numel())
       return grad_output,grad_output



class BinarizeLinear(nn.Linear):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeLinear, self).__init__(*kargs, **kwargs)

    def forward(self, input):

        if input.size(1) != 784:
            input_b=binarized(input)
        weight_b=binarized(self.weight)
        out = nn.functional.linear(input_b,weight_b)
        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1).expand_as(out)

        return out

class BinarizeConv2d(nn.Conv2d):

    def __init__(self, *kargs, **kwargs):
        super(BinarizeConv2d, self).__init__(*kargs, **kwargs)


    def forward(self, input):
        if input.size(1) != 3:
            input_b = binarized(input, quant_mode='stochastic')
        else:
            input_b=input
        weight_b=binarized(self.weight)

        out = nn.functional.conv2d(input_b, weight_b, None, self.stride,
                                   self.padding, self.dilation, self.groups)

        if not self.bias is None:
            self.bias.org=self.bias.data.clone()
            out += self.bias.view(1, -1, 1, 1).expand_as(out)

        return out
