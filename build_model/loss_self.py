
import torch
import torch.nn as nn
import torch.optim as optim




# 定义加权交叉熵损失函数，正确性待定， 20240430
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.weights = weights

    def forward(self, outputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')
        batch_size, num_classes, _, _ = outputs.size()

        # 计算每个像素的交叉熵损失
        loss = ce_loss(outputs, targets)

        # 根据标签中的信息为每个像素分配权重
        weights = torch.zeros_like(targets, dtype=torch.float32)
        for i in range(num_classes):
            weights += (targets == i).float() * self.weights[i]

        # 求加权平均损失
        weighted_loss = torch.mean(weights * loss)

        return weighted_loss

