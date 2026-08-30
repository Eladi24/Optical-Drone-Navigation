# Vendored from ignacio-rocco/ncnet (MIT license), lib/model.py -- ONLY the
# backbone-independent neighbourhood-consensus pieces (featureL2Norm,
# FeatureCorrelation, NeighConsensus, MutualMatching), kept verbatim so they can
# be diffed against upstream. The repo's own FeatureExtraction / ImMatchNet
# (ResNet-101 backbone, training harness) are NOT vendored -- STRATEGY.md Phase
# 2+ Option A2 feeds these modules our own fine-tuned DINOv2 dense patch tokens
# instead. Upstream: https://github.com/ignacio-rocco/ncnet  (MIT)
import torch
import torch.nn as nn

from conv4d import Conv4d


def featureL2Norm(feature):
    epsilon = 1e-6
    norm = torch.pow(torch.sum(torch.pow(feature, 2), 1) + epsilon, 0.5).unsqueeze(1).expand_as(feature)
    return torch.div(feature, norm)


class FeatureCorrelation(torch.nn.Module):
    def __init__(self, shape='3D', normalization=True):
        super(FeatureCorrelation, self).__init__()
        self.normalization = normalization
        self.shape = shape
        self.ReLU = nn.ReLU()

    def forward(self, feature_A, feature_B):
        if self.shape == '3D':
            b, c, h, w = feature_A.size()
            feature_A = feature_A.transpose(2, 3).contiguous().view(b, c, h * w)
            feature_B = feature_B.view(b, c, h * w).transpose(1, 2)
            feature_mul = torch.bmm(feature_B, feature_A)
            correlation_tensor = feature_mul.view(b, h, w, h * w).transpose(2, 3).transpose(1, 2)
        elif self.shape == '4D':
            b, c, hA, wA = feature_A.size()
            b, c, hB, wB = feature_B.size()
            feature_A = feature_A.view(b, c, hA * wA).transpose(1, 2)  # [b, h*w, c]
            feature_B = feature_B.view(b, c, hB * wB)                  # [b, c, h*w]
            feature_mul = torch.bmm(feature_A, feature_B)
            # indexed [batch, row_A, col_A, row_B, col_B]
            correlation_tensor = feature_mul.view(b, hA, wA, hB, wB).unsqueeze(1)

        if self.normalization:
            correlation_tensor = featureL2Norm(self.ReLU(correlation_tensor))

        return correlation_tensor


class NeighConsensus(torch.nn.Module):
    def __init__(self, use_cuda=True, kernel_sizes=[3, 3, 3], channels=[10, 10, 1], symmetric_mode=True):
        super(NeighConsensus, self).__init__()
        self.symmetric_mode = symmetric_mode
        self.kernel_sizes = kernel_sizes
        self.channels = channels
        num_layers = len(kernel_sizes)
        nn_modules = list()
        for i in range(num_layers):
            if i == 0:
                ch_in = 1
            else:
                ch_in = channels[i - 1]
            ch_out = channels[i]
            k_size = kernel_sizes[i]
            nn_modules.append(Conv4d(in_channels=ch_in, out_channels=ch_out, kernel_size=k_size, bias=True))
            nn_modules.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*nn_modules)
        if use_cuda:
            self.conv.cuda()

    def forward(self, x):
        if self.symmetric_mode:
            # apply on the input and its A-B<->B-A "transpose", then transpose the
            # second result back and add. ReLUs between layers make this different
            # from a single conv with filters+filters^T, so it is worth doing.
            x = self.conv(x) + self.conv(x.permute(0, 1, 4, 5, 2, 3)).permute(0, 1, 4, 5, 2, 3)
        else:
            x = self.conv(x)
        return x


def MutualMatching(corr4d):
    batch_size, ch, fs1, fs2, fs3, fs4 = corr4d.size()

    corr4d_B = corr4d.view(batch_size, fs1 * fs2, fs3, fs4)   # [b, k_A, i_B, j_B]
    corr4d_A = corr4d.view(batch_size, fs1, fs2, fs3 * fs4)

    corr4d_B_max, _ = torch.max(corr4d_B, dim=1, keepdim=True)
    corr4d_A_max, _ = torch.max(corr4d_A, dim=3, keepdim=True)

    eps = 1e-5
    corr4d_B = corr4d_B / (corr4d_B_max + eps)
    corr4d_A = corr4d_A / (corr4d_A_max + eps)

    corr4d_B = corr4d_B.view(batch_size, 1, fs1, fs2, fs3, fs4)
    corr4d_A = corr4d_A.view(batch_size, 1, fs1, fs2, fs3, fs4)

    corr4d = corr4d * (corr4d_A * corr4d_B)  # parentheses matter for symmetric output
    return corr4d
