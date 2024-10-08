import torch.nn.functional as F
import torch.nn as nn
import torch


class LossCombine:
    def __init__(self, ld):
        self.L1Loss = nn.L1Loss()
        self.L1GradLoss_X = nn.L1Loss()
        self.L1GradLoss_Y = nn.L1Loss()
        self.ld = ld
    def grad(self, image):
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)

        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        x_grad = F.conv2d(image, sobel_x, padding=1)
        y_grad = F.conv2d(image, sobel_y, padding = 1)
        return x_grad, y_grad

    def ssim(self, img1, img2, val_range , window_size=11, window=None, size_average=True, full=False):
        import kornia
        ssim = kornia.losses.SSIM(window_size=11,max_val=val_range,reduction='none')
        return ssim(img1, img2)
    def __call__(self, pred, true):
        loss1 = self.L1Loss(pred, true)

        x_pred, y_pred = self.grad(pred)
        x_true, y_true = self.grad(true)
        loss_x = self.L1GradLoss_X(x_pred, x_true)
        loss_y = self.L1GradLoss_Y(y_pred, y_true)

        loss_ssim = self.ssim(pred, true, val_range=1000.0 / 10.0)

        loss = self.ld * loss1 + loss_x + loss_y + loss_ssim
        return loss






