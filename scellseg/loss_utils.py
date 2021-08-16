import torch
import torch. nn as nn
import torch. nn. functional as func

def xentropy_loss(pred, true, reduction="mean"):
    """
    from hovernet
    Cross entropy loss. Assumes NHWC!
    """
    epsilon = 10e-8
    # scale preds so that the class probs of each sample sum to 1
    pred = pred / torch.sum(pred, -1, keepdim=True)  #
    # manual computation of crossentropy
    pred = torch.clamp(pred, epsilon, 1.0 - epsilon)
    loss = -torch.sum((true * torch.log(pred)), -1, keepdim=True)
    loss = loss.mean() if reduction == "mean" else loss.sum()
    return loss

def dice_loss(pred, true, smooth=1e-3):
    """
    from hovernet
    `pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC.
    """
    inse = torch.sum(pred * true , (0, 1, 2))
    l = torch.sum(pred, (0, 1, 2))
    r = torch.sum(true, (0, 1, 2))
    loss = 1.0 - (2.0 * inse + smooth) / (l + r + smooth)
    loss = torch.mean(loss)
    return loss

class TripletLossFunc(nn.Module):
    """
    from https://www.zhihu.com/question/66988664
    """
    def __init__(self, t1, t2, beta):
        super(TripletLossFunc, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.beta = beta
        return

    def forward(self, anchor, positive, negative):
        matched = torch.pow(func.pairwise_distance(anchor, positive), 2)
        mismatched = torch.pow(func.pairwise_distance(anchor, negative), 2)
        part_1 = torch.clamp(matched-mismatched, min=self.t1)
        part_2 = torch.clamp(matched, min=self.t2)
        dist_hinge = part_1 + self.beta * part_2
        loss = torch.mean(dist_hinge)
        return loss


def one_hot(y):
    '''
    from https://zhuanlan.zhihu.com/p/267660467
    y: (N)的一维tensor，值为每个样本的类别
    out: y_onehot: 转换为one_hot编码格式
    '''
    y = y.view(-1, 1)
    y_onehot = torch.FloatTensor(3, 5)

    # In your for loop
    y_onehot.zero_()
    y_onehot.scatter_(1, y, 1)
    return y_onehot