import torch
from torch import nn
import math

class Prototype(nn.Module):
    """
    Keep running average for prototype.
    """

    def __init__(self, momentum=0.1, numofclass=10):
        super(Prototype, self).__init__()
        self.num_batches_tracked = 0
        self.numofclass = numofclass
        self.momentum = momentum
        self.running_mean = None

    def forward(self, x, label):

        self.num_batches_tracked += 1
        if self.momentum is None:  # use cumulative moving average
            average_factor = 1.0 / float(self.num_batches_tracked)
        else:  # use exponential moving average
            average_factor = self.momentum

        # calculate running estimates according to classes
        batch_stat = torch.zeros((self.numofclass, x.shape[1], x.shape[2], x.shape[3])).cuda()
        batch_stat.index_add_(0, label, x)
        # In case batch size not equal to # categories
        unique_val, counts = torch.unique(label, return_counts=True)
        counts_vec = torch.zeros(self.numofclass).cuda()
        counts_vec[unique_val] = counts*1.0
        counts_vec[counts_vec==0.0] = 1.0
        batch_stat /= counts_vec.view((batch_stat.shape[0], 1,1,1))

        with torch.no_grad():
            if self.running_mean == None:
                self.running_mean = batch_stat
            else:
                self.running_mean = average_factor * batch_stat\
                    + (1 - average_factor) * self.running_mean

        return self.running_mean

class GroupNorm(nn.Module):
    """
    Applies an group normalization (pytorch default) to the input.
    """
    def __init__(self, num_groups, num_features, **kwargs):
        super().__init__()

        self.group_norm = nn.GroupNorm(num_groups, num_features, **kwargs)
        self.group_norm_t = nn.GroupNorm(num_groups, num_features, **kwargs)

#         if self.group_norm.affine==True:
#             self.group_norm.weight.data.fill_(1)
#             self.group_norm.bias.data.zero_()
#             self.group_norm_t.weight.data.fill_(1)
#             self.group_norm_t.bias.data.zero_()

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.group_norm(x)

        elif 'backward' in step:
            return self.group_norm_t(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")

class InsNorm(nn.Module):
    """
    Normalize the the input by the norm of each channel.
    """
    def forward(self, x, std=False, returnnorm=False):
        if (std==True):
            normx = torch.sqrt(torch.var(x, dim=(2,3)).reshape([x.shape[0],x.shape[1],1,1]) + 1e-8)
        else:
            normx = torch.sqrt(torch.mean(x**2, dim=(2,3)).reshape([x.shape[0],x.shape[1],1,1]) + 1e-8)
        if returnnorm==True:
            return x/normx, torch.mean(normx)
        else:
            return x/normx

class Flatten(nn.Module):
    """
    Flattens the input into vector.
    """
    def forward(self, x, step='forward'):
        if 'forward' in step:
            self.size = x.size()
            batch_size = x.size(0)
            return x.view(batch_size, -1)

        elif 'backward' in step:
            batch_size = x.size(0)
            return x.view(batch_size, *self.size[1:])

        else:
            raise ValueError("step must be 'forward' or 'backward'")

class Conv2d(nn.Module):
    """
    Applies a 2D convolution over the input. In the feedback step,
    a transposed 2D convolution is applied to the input with the same weights
    as the 2D convolution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, bias=False, last_block=False, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              bias=bias, **kwargs)

        if last_block == True or self.conv.stride[0] == 1:
            self.conv_t = nn.ConvTranspose2d(out_channels, in_channels,
                                            kernel_size, bias=bias, **kwargs)
        else:
            self.conv_t = nn.ConvTranspose2d(out_channels, in_channels,
                                            kernel_size, bias=bias, output_padding=1, **kwargs)


        n = self.conv.kernel_size[0] * self.conv.kernel_size[1] * self.conv.out_channels
        self.conv.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization
        self.conv_t.weight.data.normal_(0, math.sqrt(2. / n))  # He initialization

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.conv(x)

        elif 'backward' in step:
            return self.conv_t(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class Linear(nn.Module):
    """
    Applies a linear transform over the input. In the feedback step,
    a transposed linear transform is applied to the input with the transposed
    weights of the linear transform.
    """
    def __init__(self, in_features, out_features, bias=False, **kwargs):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear_t = nn.Linear(out_features, in_features, bias=bias)

        if(bias==True):
            self.linear.bias.data.zero_()

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.linear(x)

        elif 'backward' in step:
            return self.linear_t(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class ReLU(nn.Module):
    """
    AdaReLU: Applies AdaReLU activation over the input. In the feedback step,
    the input is pointwise multiplied through the units that were activated
    in the forward step---which are save during the forward step.
    """
    def __init__(self):
        super().__init__()
        self.state = None
        self.hidden = None

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            # Store which weights were activated
            if self.hidden is None:
                self.state = (x > 0)
            else:
                self.state = (x * self.hidden) > 0
            result = x * self.state.float()

            return result

        elif 'backward' in step:
            # Units that were activated in the forward step are passed through
            self.hidden = x
            masked_hidden = x * self.state.float()
            if unit_space is not None:
                return masked_hidden, unit_space
            else:
                return masked_hidden

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.hidden = None
        self.state = None

class resReLU(nn.Module):
    """
    AdaReLU with residual updates.
    """
    def __init__(self, res_param = 0.1):
        super().__init__()
        self.state = None
        self.hidden = None
        self.res_param = res_param

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            # Store which weights were activated
            if self.hidden is None:
                self.state = (x > 0)
            else:
                x = x + self.res_param * (self.hidden - x)
                self.state = (x * self.hidden) > 0
            result = x * self.state.float()

            return result

        elif 'backward' in step:
            # Units that were activated in the forward step are passed through
            self.hidden = x
            masked_hidden = x * self.state.float()
            if unit_space is not None:
                return masked_hidden, unit_space
            else:
                return masked_hidden

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.hidden = None
        self.state = None

class AvgPool2d(nn.Module):

    def __init__(self, kernel_size, scale_factor=10, **kwargs):
        super().__init__()

        self.avgpool = nn.AvgPool2d(kernel_size, **kwargs)
        self.upsample = nn.UpsamplingNearest2d(scale_factor=scale_factor, **kwargs)  # feedforward, before avgpool, size is 10

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return self.avgpool(x)

        elif 'backward' in step:
            return self.upsample(x)

        else:
            raise ValueError("step must be 'forward' or 'backward'")


class MaxPool2d(nn.Module):
    """
    AdaPool:
    In the feedforward pass, if the pixel in g that governs this grid is > 0, do maxpool on this grid
    In the feedforward pass, if the pixel in g that governs this grid is < 0, do min pool on this grid
    In the feedback pass, if the pixel comes from a max value, put it back to the position of the max value
    In the feedback pass, if the pixel comes from a min value, put it back to the position of the min value
    """
    def __init__(self, kernel_size, **kwargs):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size, return_indices=True, **kwargs)
        self.unpool = nn.MaxUnpool2d(kernel_size, **kwargs)

        self.hidden = None
        self.pos_state = None
        self.neg_state = None
        self.pos_ind = None
        self.neg_ind = None
        self.unpoolsize = None

    def forward(self, x, null_space=None, unit_space=None, step='forward'):
        if 'forward' in step:
            if self.hidden is None:
                self.unpoolsize = x.shape
                max_pool, self.pos_ind = self.maxpool(x)   # self.state is the indices of maxpool
                return max_pool

            else:
                max_pool, self.pos_ind = self.maxpool(x)
                min_pool, self.neg_ind = self.maxpool(-x)
                min_pool = -min_pool
                self.pos_state = (self.hidden > 0).float()
                self.neg_state = (self.hidden < 0).float()
                out = self.pos_state * max_pool + self.neg_state * min_pool
                return out

        elif 'backward' in step:
            self.hidden = x
            if ((self.pos_state is None) or (self.neg_state is None)):  # reconstruction in the first iteration, using maxpool states
                max_unpool = self.unpool(x, self.pos_ind, output_size=self.unpoolsize)
                return max_unpool
            else:
                max_unpool = self.unpool(x * self.pos_state, self.pos_ind, output_size=self.unpoolsize)
                min_unpool = self.unpool(x * self.neg_state, self.neg_ind, output_size=self.unpoolsize)
                return max_unpool + min_unpool

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def reset(self):
        self.hidden = None
        self.pos_state = None
        self.neg_state = None
        self.pos_ind = None
        self.neg_ind = None


class Bias(nn.Module):
    """
    Add a bias to the input. In the feedback step, the bias is subtracted
    from the input.
    """
    def __init__(self, size):
        super().__init__()
        self.bias = nn.Parameter(torch.Tensor(*size))
        self.bias.data.zero_()

    def forward(self, x, step='forward'):
        if 'forward' in step:
            return x + self.bias

        # no bias in backwards pass
        elif 'backward' in step:
            self.x = x
            return x

        else:
            raise ValueError("step must be 'forward' or 'backward'")

    def path_norm_loss(self, unit_space):
        return torch.mean((self.x * self.bias - unit_space * self.bias)**2)


class Dropout(nn.Module):
    """
    Performs dropout regularization to the input. In the feedback step, the
    same dropout transformation is applied in the backwards step.
    """
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x, step='forward', training=False):
        if 'forward' in step:
            if training:
                self.dropout = (torch.rand_like(x) > self.p).float() / self.p
                return x * self.dropout
            else:
                return x

        elif 'backward' in step:
            if training:
                return x * self.dropout
            else:
                return x

        else:
            raise ValueError("step must be 'forward' or 'backward'")
