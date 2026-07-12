from typing import Dict, Any, List, Union, Optional
from math import ceil
import os

import torch
import torch.nn as nn
import torchvision.models._utils as _utils
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import nms
import yaml

from .download import get_asset

class RetinaFace(nn.Module):
    def __init__(self, model_name: str):
        """
        RetinaFace face detector that self-provisions its weights and config.

        The backbone structure is described by the asset's config under a
        ``backbone`` section:
            - 'name': Backbone model ('mobilenet' or 'resnet50').
            - 'return_layers': Which backbone layers the FPN taps.
            - 'in_channel' / 'out_channel': FPN channel widths.

        The full detector checkpoint (body + FPN + SSH + heads) is loaded into
        this module.

        Args:
            model_name (str): Detector asset name in the release manifest
                (e.g. ``retinaface_mobilenetv1`` or ``retinaface_resnet50``).
        """
        super(RetinaFace, self).__init__()
        self.model_name = model_name
        asset_info = get_asset(model_name)
        with open(os.path.join(asset_info['path'], asset_info['contents']['config']), "r", encoding="utf-8") as f:
            cfgs = yaml.safe_load(f)
        # Tolerate both the nested ({'backbone': {...}}) and legacy flat layouts.
        bb = cfgs.get('backbone', cfgs)
        name = bb['name']
        if name == 'mobilenet':
            backbone = MobileNetV1()
        elif name == 'resnet50':
            backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"Unsupported backbone for RetinaFace: {name}. Supported backbones are 'mobilenet' and 'resnet50'.")

        self.body = _utils.IntermediateLayerGetter(backbone, bb['return_layers'])
        in_channels_stage2 = bb['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = bb['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)
        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=out_channels)
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=out_channels)
        self.LandmarkHead = self._make_landmark_head(fpn_num=3, inchannels=out_channels)

        # Load the FULL detector checkpoint (body + FPN + SSH + heads).
        ckpt = self._clean_ckpt(torch.load(
            os.path.join(asset_info['path'], asset_info['contents']['weight']),
            map_location="cpu",
            weights_only=True,
        ))
        self.load_state_dict(ckpt, strict=True)
        self.eval()

    @staticmethod
    def _clean_ckpt(ckpt: Dict[str, Any]) -> Dict[str, Any]:
        f = lambda x: x.split('module.', 1)[-1] if x.startswith('module.') else x
        if "state_dict" in ckpt:
            return {f(k): v for k, v in ckpt["state_dict"].items()}
        return {f(k): v for k, v in ckpt.items()}

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_landmark_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        landmarkhead = nn.ModuleList()
        for i in range(fpn_num):
            landmarkhead.append(LandmarkHead(inchannels,anchor_num))
        return landmarkhead

    def forward(self, 
                imgs: torch.Tensor, 
                need_decode: bool = True, 
                need_nms: bool = True, 
                conf_thresh: float = 0.8, 
                iou_thresh: float = 0.4,
                variances: List[float] = [0.1, 0.2],
                min_sizes: List[List[int]] = [[16, 32], [64, 128], [256, 512]],
                steps: List[int] = [8, 16, 32],
                clip: bool = False) -> List[Union[torch.Tensor, None]]:
        """
        Full forward pass: run the backbone, FPN, SSH and detection heads, then optionally
        decode the predictions and apply NMS to produce final face detections.

        Args:
            imgs (torch.Tensor): Input image tensor of shape BCHW, range [-128, 127], BGR, torch.float32
            need_decode (bool): Whether to decode the raw predictions into bounding boxes and landmarks. Default is True. 
                The output will be differentiable w.r.t. the input images no matter the value of need_decode.
            need_nms (bool): Whether to apply Non-Maximum Suppression (NMS) to the decoded predictions. Default is True. 
                If True, the output won't be differentiable w.r.t. the input images. Only used if need_decode is True.
            conf_thresh (float): Confidence threshold for filtering detections. Default is 0.8. Only used if need_decode is True.
            iou_thresh (float): IoU threshold for NMS. Default is 0.4. Only used if need_decode is True.
            variances (List[float]): Variances for decoding the bounding boxes and landmarks. Default is [0.1, 0.2].
            min_sizes (List[List[int]]): Minimum sizes for the anchor boxes at each feature map level. Default is [[16, 32], [64, 128], [256, 512]].
            steps (List[int]): Steps for the anchor boxes at each feature map level. Default is [8, 16, 32].
            clip (bool): Whether to clip the decoded bounding boxes to be within [0, 1]. Default is False.
            

        Returns:
            List[torch.Tensor]: Always a list of length B (one tensor per image). Each tensor has 16 columns:
                    [box(4), cls(2), ldm(10)]
                - need_decode is False: per image (N, 16), raw undecoded outputs
                    [bbox_reg(4), cls_logits(2), ldm_reg(10)].
                - need_decode is True, need_nms is False: per image (N, 16), decoded
                    [x1, y1, x2, y2, cls_prob(2), ldm1_x, ldm1_y, ..., ldm5_x, ldm5_y] in pixels,
                    where cls_prob are softmax probabilities (col 5 is the face probability).
                - need_decode is True, need_nms is True: per image (K_b, 16), same column layout
                    after confidence thresholding + NMS (K_b varies per image, hence a list).
        """
        out = self.body(imgs)
        fpn = self.fpn(out)
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)], dim=1)
        ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if not need_decode:
            raw = torch.cat([bbox_regressions, classifications, ldm_regressions], dim=-1)  # (B, N, 16)
            return list(raw.unbind(dim=0))

        B, H, W = imgs.shape[0], imgs.shape[2], imgs.shape[3]
        priors = self._get_prior(H, W, steps, min_sizes, clip=clip, device=imgs.device)
        scale_box = imgs.new_tensor([W, H, W, H])
        scale_ldm = imgs.new_tensor([W, H] * 5)

        # Decode the whole batch at once (priors broadcast over the batch dim).
        boxes = self._decode_bbox(bbox_regressions, priors, variances) * scale_box   # (B, N, 4)
        probs = F.softmax(classifications, dim=-1)                                    # (B, N, 2)
        landms = self._decode_ldmk(ldm_regressions, priors, variances) * scale_ldm   # (B, N, 10)
        decoded_results = torch.cat([boxes, probs, landms], dim=-1)                   # (B, N, 16)

        if not need_nms:
            return list(decoded_results.unbind(dim=0))

        return [self._nms(decoded_results[b], conf_thresh, iou_thresh) for b in range(B)]
            
    @staticmethod
    def _nms(dets: torch.Tensor, conf_thresh: float, iou_thresh: float) -> Optional[torch.Tensor]:
        # dets: (N, 16) -> [box(4), cls_prob(2), ldm(10)]; column 5 is the face probability.
        mask = dets[:, 5] > conf_thresh
        dets = dets[mask]
        if dets.shape[0] == 0:
            return None
        keep = nms(dets[:, :4], dets[:, 5], iou_thresh)
        return dets[keep]
        
    @staticmethod
    def _get_prior(h: int, w: int, steps: List[int], min_sizes: List[List[int]], clip: bool, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
            anchors = []
            for k, step in enumerate(steps):
                min_sizes_k = min_sizes[k]
                num_min = len(min_sizes_k)
                fh, fw = ceil(h / step), ceil(w / step)

                ys, xs = torch.meshgrid(
                    torch.arange(fh, device=device, dtype=torch.float32),
                    torch.arange(fw, device=device, dtype=torch.float32),
                    indexing='ij',
                )
                cx = ((xs + 0.5) * step / w).unsqueeze(-1).expand(fh, fw, num_min)
                cy = ((ys + 0.5) * step / h).unsqueeze(-1).expand(fh, fw, num_min)
                ms = torch.tensor(min_sizes_k, device=device, dtype=torch.float32)
                s_kx = (ms / w).view(1, 1, num_min).expand(fh, fw, num_min)
                s_ky = (ms / h).view(1, 1, num_min).expand(fh, fw, num_min)
                anchors.append(torch.stack([cx, cy, s_kx, s_ky], dim=-1).reshape(-1, 4))
            output = torch.cat(anchors, dim=0)
            if clip:
                output.clamp_(max=1., min=0.)
            return output
        
    @staticmethod
    def _decode_bbox(loc: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
        """
        Decode bbox regressions into corner-form boxes (x1, y1, x2, y2), normalized.

        Ellipsis indexing keeps this valid for both (N, 4) and batched (B, N, 4) inputs;
        ``priors`` is (N, 4) and broadcasts against the leading batch dim.
        """
        centers = priors[..., :2] + loc[..., :2] * variances[0] * priors[..., 2:]
        sizes = priors[..., 2:] * torch.exp(loc[..., 2:] * variances[1])
        xy_min = centers - sizes / 2
        xy_max = xy_min + sizes
        return torch.cat([xy_min, xy_max], dim=-1)

    @staticmethod
    def _decode_ldmk(ldmk: torch.Tensor, priors: torch.Tensor, variances: List[float]) -> torch.Tensor:
        """
        Decode landmark regressions into 5 (x, y) points, normalized -> shape (..., 10).
        Works for both (N, 10) and batched (B, N, 10) inputs (priors broadcast over batch).
        """
        return torch.cat([
            priors[..., :2] + ldmk[..., i:i + 2] * variances[0] * priors[..., 2:]
            for i in range(0, 10, 2)
        ], dim=-1)

class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class LandmarkHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(LandmarkHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*10,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 10)

def conv_bn(inp, oup, stride = 1, leaky = 0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_bn_no_relu(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
    )

def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope=leaky, inplace=True)
    )

def conv_dw(inp, oup, stride, leaky=0.1):
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.LeakyReLU(negative_slope= leaky,inplace=True),
    )

class SSH(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SSH, self).__init__()
        assert out_channel % 4 == 0
        leaky = 0
        if (out_channel <= 64):
            leaky = 0.1
        self.conv3X3 = conv_bn_no_relu(in_channel, out_channel//2, stride=1)

        self.conv5X5_1 = conv_bn(in_channel, out_channel//4, stride=1, leaky = leaky)
        self.conv5X5_2 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

        self.conv7X7_2 = conv_bn(out_channel//4, out_channel//4, stride=1, leaky = leaky)
        self.conv7x7_3 = conv_bn_no_relu(out_channel//4, out_channel//4, stride=1)

    def forward(self, input):
        conv3X3 = self.conv3X3(input)

        conv5X5_1 = self.conv5X5_1(input)
        conv5X5 = self.conv5X5_2(conv5X5_1)

        conv7X7_2 = self.conv7X7_2(conv5X5_1)
        conv7X7 = self.conv7x7_3(conv7X7_2)

        out = torch.cat([conv3X3, conv5X5, conv7X7], dim=1)
        out = F.relu(out)
        return out

class FPN(nn.Module):
    def __init__(self,in_channels_list,out_channels):
        super(FPN,self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0], out_channels, stride = 1, leaky = leaky)
        self.output2 = conv_bn1X1(in_channels_list[1], out_channels, stride = 1, leaky = leaky)
        self.output3 = conv_bn1X1(in_channels_list[2], out_channels, stride = 1, leaky = leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky = leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky = leaky)

    def forward(self, input):
        # names = list(input.keys())
        input = list(input.values())

        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        out = [output1, output2, output3]
        return out



class MobileNetV1(nn.Module):
    def __init__(self):
        super(MobileNetV1, self).__init__()
        self.stage1 = nn.Sequential(
            conv_bn(3, 8, 2, leaky = 0.1),    # 3
            conv_dw(8, 16, 1),   # 7
            conv_dw(16, 32, 2),  # 11
            conv_dw(32, 32, 1),  # 19
            conv_dw(32, 64, 2),  # 27
            conv_dw(64, 64, 1),  # 43
        )
        self.stage2 = nn.Sequential(
            conv_dw(64, 128, 2),  # 43 + 16 = 59
            conv_dw(128, 128, 1), # 59 + 32 = 91
            conv_dw(128, 128, 1), # 91 + 32 = 123
            conv_dw(128, 128, 1), # 123 + 32 = 155
            conv_dw(128, 128, 1), # 155 + 32 = 187
            conv_dw(128, 128, 1), # 187 + 32 = 219
        )
        self.stage3 = nn.Sequential(
            conv_dw(128, 256, 2), # 219 +3 2 = 241
            conv_dw(256, 256, 1), # 241 + 64 = 301
        )
        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(256, 1000)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg(x)
        # x = self.model(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        return x

