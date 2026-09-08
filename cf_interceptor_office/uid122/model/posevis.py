from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn

ROOM_X, ROOM_Y = 18.0, 7.6
NX, NY, NYAW = 36, 16, 24
CELL_W, CELL_H = ROOM_X / NX, ROOM_Y / NY
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _conv3(i, o, s=1):
    return nn.Conv2d(i, o, 3, s, 1, bias=False)


class _Block(nn.Module):
    expansion = 1

    def __init__(self, inp, out, stride=1, down=None):
        super().__init__()
        self.conv1 = _conv3(inp, out, stride); self.bn1 = nn.BatchNorm2d(out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3(out, out); self.bn2 = nn.BatchNorm2d(out)
        self.downsample = down

    def forward(self, x):
        idt = x
        o = self.relu(self.bn1(self.conv1(x)))
        o = self.bn2(self.conv2(o))
        if self.downsample is not None:
            idt = self.downsample(x)
        return self.relu(o + idt)


class _ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make(64, 2)
        self.layer2 = self._make(128, 2, 2)
        self.layer3 = self._make(256, 2, 2)
        self.layer4 = self._make(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Identity()

    def _make(self, planes, blocks, stride=1):
        down = None
        if stride != 1 or self.inplanes != planes:
            down = nn.Sequential(nn.Conv2d(self.inplanes, planes, 1, stride, bias=False),
                                 nn.BatchNorm2d(planes))
        layers = [_Block(self.inplanes, planes, stride, down)]
        self.inplanes = planes
        layers += [_Block(planes, planes) for _ in range(blocks - 1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return torch.flatten(self.avgpool(x), 1)


class PoseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ResNet18()
        self.cell = nn.Linear(512, NX * NY)
        self.off = nn.Linear(512, 2)
        self.yaw = nn.Linear(512, NYAW)
        self.yawres = nn.Linear(512, 2)
        self.logsig = nn.Linear(512, 2)

    def forward(self, x):
        h = self.backbone(x)
        return (self.cell(h), torch.sigmoid(self.off(h)), self.yaw(h),
                self.yawres(h), self.logsig(h).clamp(-4.0, 2.0))


class VisualPose:

    def __init__(self, weights_path, threads: int = 1):
        torch.set_num_threads(max(1, int(threads)))
        self.net = PoseNet()
        sd = torch.load(weights_path, map_location="cpu", weights_only=True)
        self.net.load_state_dict({k: v.float() for k, v in sd.items()})
        self.net.eval()
        self._sig = None
        self._last = None
        with torch.inference_mode():
            self.net(torch.zeros(1, 3, 128, 128))

    def reset(self):
        self._sig, self._last = None, None

    @staticmethod
    def _signature(rgb):
        s = rgb[::32, ::32]
        return (float(s.sum()), float(s[::2, ::2].sum()))

    def predict(self, rgb):
        rgb = np.asarray(rgb, dtype=np.float32)
        if rgb.shape != (256, 256, 3):
            return None


        sig = self._signature(rgb)
        if sig == self._sig and self._last is not None:
            return self._last[0], self._last[1], self._last[2], False
        self._sig = sig
        from PIL import Image
        im = Image.fromarray((np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8))
        im = im.resize((128, 128), Image.Resampling.BILINEAR)
        a = (np.asarray(im, dtype=np.float32) / 255.0 - _MEAN) / _STD
        x = torch.from_numpy(np.transpose(a, (2, 0, 1))[None].copy())
        with torch.inference_mode():
            lc, lo, ly, lyr, ls = self.net(x)
        p = torch.softmax(lc[0], 0)
        k = int(p.argmax())
        conf = float(p[k])
        ox, oy = float(lo[0, 0]), float(lo[0, 1])
        xy = np.array([(k % NX) * CELL_W + ox * CELL_W,
                       (k // NX) * CELL_H + oy * CELL_H], dtype=np.float64)
        yaw = float(math.atan2(float(lyr[0, 0]), float(lyr[0, 1])))
        self._last = (xy, yaw, conf)
        return xy, yaw, conf, True
