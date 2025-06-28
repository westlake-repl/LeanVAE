import os
import random
import shutil
import sys
from datetime import datetime
import math
import numpy as np
import pywt
import torch
from torch.nn import Module
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn

_PERSISTENT = False


class Patcher(torch.nn.Module):
    def __init__(self, rescale = True):
        super().__init__()
        self.register_buffer(
            "wavelets", torch.tensor([0.7071067811865476, 0.7071067811865476]), persistent=_PERSISTENT
        )
        self.register_buffer(
            "_arange",
            torch.arange(2),
            persistent=_PERSISTENT,
        )

        self.rescale = rescale
        for param in self.parameters():
            param.requires_grad = False
    

    
    def _2ddwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        x = x.squeeze(2)

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        x = F.pad(x, pad=(n - 2, n - 1, n - 2, n - 1), mode=mode).to(dtype)
        xl = F.conv2d(x, hl.unsqueeze(2), groups=g, stride=(1, 2))
        xh = F.conv2d(x, hh.unsqueeze(2), groups=g, stride=(1, 2))
        xll = F.conv2d(xl, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xlh = F.conv2d(xl, hh.unsqueeze(3), groups=g, stride=(2, 1))
        xhl = F.conv2d(xh, hl.unsqueeze(3), groups=g, stride=(2, 1))
        xhh = F.conv2d(xh, hh.unsqueeze(3), groups=g, stride=(2, 1))

        out = torch.cat([xll, xlh, xhl, xhh], dim=1)
        if rescale:
            out = out * 2
        return out.unsqueeze(2)
    
    def _3ddwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets

        n = h.shape[0]
        g = x.shape[1]
        hl = h.flip(0).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        # Handles temporal axis.
        x = F.pad(
            x, pad=(max(0, n - 2), n - 1, n - 2, n - 1, n - 2, n - 1), mode=mode
        ).to(dtype)
        xl = F.conv3d(x, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))
        xh = F.conv3d(x, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1))

        # Handles spatial axes.
        xll = F.conv3d(xl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xlh = F.conv3d(xl, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhl = F.conv3d(xh, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))
        xhh = F.conv3d(xh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1))

        xlll = F.conv3d(xll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xllh = F.conv3d(xll, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhl = F.conv3d(xlh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xlhh = F.conv3d(xlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhll = F.conv3d(xhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhlh = F.conv3d(xhl, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhl = F.conv3d(xhh, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))
        xhhh = F.conv3d(xhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2))

        out = torch.cat([xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh], dim=1)
        if rescale:
            out = out * (2 * torch.sqrt(torch.tensor(2.0)))
        return out

    def forward(self, x):
        if x.shape[2] > 1:
            if x.shape[2] % 2 == 1:
                xi, xv = torch.split(x, [1, x.shape[2] - 1], dim=2)
                xi = self._2ddwt(xi, rescale=self.rescale)
                xv = self._3ddwt(xv, rescale=self.rescale)
                return xi, xv
            else:
                xv = self._3ddwt(x, rescale=self.rescale)
                return None, xv

        return (self._2ddwt(x, rescale=self.rescale), None)


class UnPatcher(torch.nn.Module):

    def __init__(self, rescale = True):
        super().__init__()
        self.register_buffer(
            "wavelets", torch.tensor([0.7071067811865476, 0.7071067811865476]), persistent=_PERSISTENT
        )
        self.register_buffer(
            "_arange",
            torch.arange(2),
            persistent=_PERSISTENT,
        )
        self.rescale = rescale
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        xi, xv = x
        if xi is not None and xv is not None:
            xi = self._2didwt(xi, rescale=self.rescale)
            xv = self._3didwt(xv, rescale=self.rescale)
            return torch.cat([xi.unsqueeze(2), xv], dim=2)
        elif xv is None and xi is not None:
            return self._2didwt(xi, rescale=self.rescale)
        elif xv is not None and xi is None:
            return self._3didwt(xv, rescale=self.rescale)

    def _2didwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]
        x = x.squeeze(2)

        g = x.shape[1] // 4
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hh = hh.to(dtype=dtype)
        hl = hl.to(dtype=dtype)

        xll, xlh, xhl, xhh = torch.chunk(x.to(dtype), 4, dim=1)

        # Inverse transform.
        yl = torch.nn.functional.conv_transpose2d(
            xll, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yl += torch.nn.functional.conv_transpose2d(
            xlh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh = torch.nn.functional.conv_transpose2d(
            xhl, hl.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        yh += torch.nn.functional.conv_transpose2d(
            xhh, hh.unsqueeze(3), groups=g, stride=(2, 1), padding=(n - 2, 0)
        )
        y = torch.nn.functional.conv_transpose2d(
            yl, hl.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )
        y += torch.nn.functional.conv_transpose2d(
            yh, hh.unsqueeze(2), groups=g, stride=(1, 2), padding=(0, n - 2)
        )

        if rescale:
            y = y / 2
        return y
    
    def _3didwt(self, x, mode="reflect", rescale=False):
        dtype = x.dtype
        h = self.wavelets
        n = h.shape[0]

        g = x.shape[1] // 8  # split into 8 spatio-temporal filtered tesnors.
        hl = h.flip([0]).reshape(1, 1, -1).repeat([g, 1, 1])
        hh = (h * ((-1) ** self._arange)).reshape(1, 1, -1).repeat(g, 1, 1)
        hl = hl.to(dtype=dtype)
        hh = hh.to(dtype=dtype)

        xlll, xllh, xlhl, xlhh, xhll, xhlh, xhhl, xhhh = torch.chunk(x, 8, dim=1)

        # Height height transposed convolutions.
        xll = F.conv_transpose3d(
            xlll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xll += F.conv_transpose3d(
            xllh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xlh = F.conv_transpose3d(
            xlhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xlh += F.conv_transpose3d(
            xlhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xhl = F.conv_transpose3d(
            xhll, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhl += F.conv_transpose3d(
            xhlh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        xhh = F.conv_transpose3d(
            xhhl, hl.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )
        xhh += F.conv_transpose3d(
            xhhh, hh.unsqueeze(2).unsqueeze(3), groups=g, stride=(1, 1, 2)
        )

        # Handles width transposed convolutions.
        xl = F.conv_transpose3d(
            xll, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xl += F.conv_transpose3d(
            xlh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xh = F.conv_transpose3d(
            xhl, hl.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )
        xh += F.conv_transpose3d(
            xhh, hh.unsqueeze(2).unsqueeze(4), groups=g, stride=(1, 2, 1)
        )

        # Handles time axis transposed convolutions.
        x = F.conv_transpose3d(
            xl, hl.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )
        x += F.conv_transpose3d(
            xh, hh.unsqueeze(3).unsqueeze(4), groups=g, stride=(2, 1, 1)
        )

        if rescale:
            x = x / (2 * torch.sqrt(torch.tensor(2.0)))
        return x

