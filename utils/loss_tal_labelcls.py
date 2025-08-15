#多标签分类版本
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.general import xywh2xyxy
from utils.metrics import bbox_iou
from utils.tal.anchor_generator import dist2bbox, make_anchors, bbox2dist
from utils.tal.assigner import TaskAlignedAssigner
from utils.torch_utils import de_parallel


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps


class VarifocalLoss(nn.Module):
    # Varifocal loss by Zhang et al. https://arxiv.org/abs/2008.13367
    def __init__(self):
        super().__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):
        weight = alpha * pred_score.sigmoid().pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy_with_logits(pred_score.float(), gt_score.float(),
                                                       reduction="none") * weight).sum()
        return loss


class FocalLoss(nn.Module):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = "none"  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss


# class AsymmetricLoss(nn.Module):
#     def __init__(self, gamma_neg=4, gamma_pos=0, margin=0.05, eps=1e-8):
#         super().__init__()
#         self.gamma_neg = gamma_neg
#         self.gamma_pos = gamma_pos
#         self.margin = margin
#         self.eps = eps
#
#     def forward(self, inputs, targets):
#         # inputs: raw logits, targets: binary labels
#         inputs = torch.clamp(inputs, min=-10, max=10)
#         preds = torch.sigmoid(inputs)
#
#         # Apply probability shifting only to negative targets
#         preds_neg_shifted = (preds - self.margin).clamp(min=0)
#
#         # Log-safe operations
#         log_preds = torch.log(preds.clamp(min=self.eps))
#         log_1_minus_preds = torch.log((1 - preds_neg_shifted).clamp(min=self.eps))
#
#         # Positive and negative losses
#         loss_pos = targets * log_preds * torch.pow(1 - preds, self.gamma_pos)
#         loss_neg = (1 - targets) * log_1_minus_preds * torch.pow(preds_neg_shifted, self.gamma_neg)
#
#         loss = - (loss_pos + loss_neg)
#         return loss.sum() / inputs.size(0)  # mean over batch


# import torch
# import torch.nn as nn
#
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=0, margin=0.05, eps=1e-8):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin = margin
        self.eps = eps

    def forward(self, inputs, targets):
        # inputs: raw logits, shape [B, C]
        # targets: binary labels, shape [B, C]
        inputs = torch.clamp(inputs, min=-10, max=10)
        preds = torch.sigmoid(inputs)

        # Apply probability shifting for negative class
        preds_neg_shifted = (preds - self.margin).clamp(min=0)

        # Log-safe
        log_preds = torch.log(preds.clamp(min=self.eps))
        log_1_minus_preds = torch.log((1 - preds_neg_shifted).clamp(min=self.eps))

        # ASL loss for pos and neg
        loss_pos = targets * log_preds * torch.pow(1 - preds, self.gamma_pos)
        loss_neg = (1 - targets) * log_1_minus_preds * torch.pow(preds_neg_shifted, self.gamma_neg)

        # Final loss: mean over all elements (like BCEWithLogitsLoss(reduction='mean'))
        loss = - (loss_pos + loss_neg)
        return loss.mean()




class BboxLoss(nn.Module):
    def __init__(self, reg_max, use_dfl=False):
        super().__init__()
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):
        # iou loss
        bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])  # (b, h*w, 4)
        pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).view(-1, 4)
        target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).view(-1, 4)
        bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

        iou = bbox_iou(pred_bboxes_pos, target_bboxes_pos, xywh=False, CIoU=True)
        loss_iou = 1.0 - iou

        loss_iou *= bbox_weight
        loss_iou = loss_iou.sum() / target_scores_sum

        # dfl loss
        if self.use_dfl:
            dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
            pred_dist_pos = torch.masked_select(pred_dist, dist_mask).view(-1, 4, self.reg_max + 1)
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
            target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).view(-1, 4)
            loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl, iou

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1),
                                     reduction="none").view(target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)


class Compute_loss_label:
    # Compute losses
    def __init__(self, model, use_dfl=True):
        device = next(model.parameters()).device  # get model device
        h = model.hyp  # hyperparameters

        # # # Define criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([h["cls_pw"]], device=device), reduction='none')
        # #
        # # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3
        # self.cp, self.cn = smooth_BCE(eps=h.get("label_smoothing", 0.0))  # positive, negative BCE targets

        # # Focal loss
        # g = h["fl_gamma"]  # focal loss gamma
        # if g > 0:
        #     BCEcls = FocalLoss(BCEcls, g)

        m = de_parallel(model).model[-1]  # Detect() module
        self.balance = {3: [4.0, 1.0, 0.4]}.get(m.nl, [4.0, 1.0, 0.25, 0.06, 0.02])  # P3-P7
        self.BCEcls = BCEcls
        self.hyp = h
        self.stride = m.stride  # model strides
        self.nc = m.nc  # number of classes
        self.nl = m.nl  # number of layers
        self.no = m.no
        self.reg_max = m.reg_max
        self.device = device
        self.BCEloss = nn.BCEWithLogitsLoss()
        # self.ASLloss = AsymmetricLoss()
        # BCEcls_label = nn.BCEWithLogitsLoss(reduction='none')  # 不要再加 pos_weight，否则形状不匹配
        # self.Focal = FocalLoss(BCEcls_label, gamma=2.0, alpha=0.25)

        self.assigner = TaskAlignedAssigner(topk=int(os.getenv('YOLOM', 10)),
                                            num_classes=self.nc,
                                            alpha=float(os.getenv('YOLOA', 0.5)),
                                            beta=float(os.getenv('YOLOB', 6.0)))
        self.bbox_loss = BboxLoss(m.reg_max - 1, use_dfl=use_dfl).to(device)
        self.proj = torch.arange(m.reg_max).float().to(device)  # / 120.0
        self.use_dfl = use_dfl

    def preprocess(self, targets, batch_size, scale_tensor):
        if targets.shape[0] == 0:
            out = torch.zeros(batch_size, 0, 5, device=self.device)
        else:
            i = targets[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            # print(f"Debug: Maximum number of targets in a single image: {counts.max().item()}")
            out = torch.zeros(batch_size, counts.max(), 5, device=self.device)

            # print("Debug: Initialized out shape:", out.shape)
            # print("Debug: targets shape inside preprocess:", targets.shape)
            # print("Debug: targets data inside preprocess (first 5 rows):\n", targets[:5])
            for j in range(batch_size):
                matches = i == j
                n = matches.sum()

                if n:
                    # print(f"Debug: Processing image {j}, matches count: {n}")
                    # print(f"Debug: out[{j}, :{n}] shape:", out[j, :n].shape)
                    # print(f"Debug: targets[matches, 1:] shape:", targets[matches, 1:].shape)
                    # print("Debug: targets[matches, 1:] data (first 5 rows):\n", targets[matches, 1:][:5])

                    out[j, :n] = targets[matches, 1:]
            out[..., 1:5] = xywh2xyxy(out[..., 1:5].mul_(scale_tensor))
            # print("Debug: After coordinate conversion, out shape:", out.shape)
            # print("Debug: out data (first 5 rows):\n", out[:5])
        return out


    def __call__(self, p, label_pred, targets, img=None, epoch=0):

        loss = torch.zeros(1, device=self.device)  # 初始化损失值，包含 box, cls, dfl, category（类别标签分类损失）
        feats = p[1] if isinstance(p, tuple) else p  # 提取特征图
        pred_distri, pred_scores = torch.cat([xi.view(feats[0].shape[0], self.no, -1) for xi in feats], 2).split(
            (self.reg_max * 4, self.nc), 1)  # 分割预测结果为分布值和分类分数

        dtype = pred_scores.dtype  # 获取数据类型
        batch_size, grid_size = pred_scores.shape[:2]  # 获取批次大小和网格数量
        imgsz = torch.tensor(feats[0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]  # 图像尺寸（高，宽）


        # 处理 targets，分离类别和边界框数据
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])  # 预处理 targets
        # 打印第一个样本的 targets 信息
        # print(f"First sample targets: {targets[0]}")  # 打印第一个样本的标注信息

        gt_labels, gt_bboxes = targets.split((1, 4), 2)  # 分离类别标签（cls）和边界框（xyhw）
        # print(f"gt_labels[0]: {gt_labels[0]}")  # 打印第一个样本的类别标签
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0)  # 生成有效目标的掩码

        gt_labels_onehot = torch.zeros(label_pred.shape, device=self.device)  # 初始化 one-hot 张量

        # for i in range(batch_size):
        #     # 临时将 mask_gt 转换为布尔类型以进行索引
        #     valid_labels = gt_labels[i, mask_gt[i].squeeze(-1).to(torch.bool)].long()
        #
        #     # 确保 valid_labels 是一维张量
        #     valid_labels = valid_labels.view(-1)  # 转换为一维
        #
        #     # 生成 one-hot 编码
        #     gt_labels_onehot[i].scatter_(0, valid_labels, 1)
        #
        # # 确保 one-hot 编码为二值
        # gt_labels_onehot = gt_labels_onehot.clamp(0, 1)

        # 获取所有有效的标签和样本索引
        batch_indices, target_indices = torch.where(mask_gt.squeeze(-1))  # 找到有效目标的样本和目标索引
        valid_labels = gt_labels[batch_indices, target_indices].long().view(-1)

        # 生成 one-hot 编码
        gt_labels_onehot[batch_indices, valid_labels] = 1

        # # 打印 batch0 的 gt_labels_onehot
        # print(f"batch0 gt_labels_onehot: {gt_labels_onehot[0]}")  # 打印 batch0 的 one-hot 编码

        # 使用 BCE 损失计算类别标签分类损失
        loss[0] = self.BCEloss(label_pred, gt_labels_onehot)

        # 设置不同损失项的权重
        loss[0] *= 1

        return loss.sum() * batch_size, loss.detach()
