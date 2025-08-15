# ——*——code:UTF-8——*——
# Author : airy
# DATA : 2023/1/17 上午8:49
import torch


# class _GradientScalarLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.weight = weight
#         return input.view_as(input)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         return ctx.weight*grad_input, None
#
# gradient_scalar = _GradientScalarLayer.apply

#记下来
class _GradientScalarLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.weight = weight
        # 为什么要用clone，避免影响其他损失的反向传播
        # 原先的克隆，可能会存在一个内存重排现在，即克隆有一定的随机性，从而导致每次训练结果不一致
        return input.contiguous().view_as(input)  # 加入contiguous确保了克隆后张量的连续性以及确定性

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()  # We'll keep the clone here for now
        return ctx.weight * grad_input, None

# class _GradientScalarLayer(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, weight):
#         ctx.weight = weight
#         return input  # 直接返回输入，不做任何形状操作
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output * ctx.weight  # 直接进行梯度缩放
#         return grad_input, None  # 返回缩放后的梯度


gradient_scalar = _GradientScalarLayer.apply


class GradientScalarLayer(torch.nn.Module):
    def __init__(self, weight):
        super(GradientScalarLayer, self).__init__()
        self.weight = weight

    def forward(self, input):
        return gradient_scalar(input, self.weight)
