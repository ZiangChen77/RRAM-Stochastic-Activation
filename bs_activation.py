import torch

class ReLU(torch.autograd.Function):
    # @staticmethod
    def forward(ctx, x):
         result = x*(x>0)
         ctx.save_for_backward(x>0)
         return result
    
    # @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
    
class ReLU_T(torch.autograd.Function):
    # @staticmethod
    def forward(ctx, x):
         result = x.clamp(min=0.0,max=1.0)
         ctx.save_for_backward((x>0)*(x<1.0))
         return result
    
    # @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result

class ReLU_BS(torch.autograd.Function):
    # @staticmethod
    def forward(ctx, x):
         result = x.clamp(min=0.0,max=1.0)
         ctx.save_for_backward((x>0)*(x<1.0))
         return result.bernoulli()*1.0
    
    # @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result


class Sigmoid(torch.autograd.Function):
    # @staticmethod
    def forward(ctx, x):
         result = 1.0/(1.0+(-4.0*x).exp())
         ctx.save_for_backward(result)
         return result
    
    # @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * (4.0*result * (1.0-result))

class Sigmoid_BS(torch.autograd.Function):
    # @staticmethod
    def forward(ctx, x):
         result = 1.0/(1.0+(-4.0*x).exp())
         ctx.save_for_backward(result)
         return result.bernoulli()
    
    # @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        grad_input = grad_output * (4.0*result * (1.0-result)).bernoulli()
        return grad_input
    
class CrossEntropy(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, inputs, target):
        input_softmax=inputs.softmax(dim=1)
        ctx.save_for_backward(input_softmax, target)
        return input_softmax, (-input_softmax.log() * target).sum()

    @staticmethod
    def backward(ctx, grad_output, loss_grad):
        input_softmax, target = ctx.saved_tensors
        grad_input = (input_softmax - target)
        return grad_input, None

class CrossEntropy_BS(torch.autograd.Function):  
    @staticmethod
    def forward(ctx, inputs, target):
        input_softmax=inputs.softmax(dim=1)
        ctx.save_for_backward(input_softmax, target)
        return input_softmax, (-input_softmax.log() * target).sum()

    @staticmethod
    def backward(ctx, grad_output, loss_grad):
        input_softmax, target = ctx.saved_tensors
        grad_input = (input_softmax.bernoulli() - target)
        return grad_input, None