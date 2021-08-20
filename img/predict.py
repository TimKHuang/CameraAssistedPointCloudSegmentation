"""
Code adapted from
https://github.com/NVIDIA/semantic-segmentation/blob/sdcnet/demo.py

Source License
# Copyright (C) 2019 NVIDIA Corporation. Yi Zhu, Karan Sapra, Fitsum A. Reda, Kevin J. Shih, Shawn Newsam, Andrew Tao and Bryan Catanzaro.
# All rights reserved. 
# Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

# Permission to use, copy, modify, and distribute this software and its documentation 
# for any non-commercial purpose is hereby granted without fee, provided that the above 
# copyright notice appear in all copies and that both that copyright notice and this 
# permission notice appear in supporting documentation, and that the name of the author 
# not be used in advertising or publicity pertaining to distribution of the software 
# without specific, written prior permission.

# THE AUTHOR DISCLAIMS ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING ALL 
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR ANY PARTICULAR PURPOSE. 
# IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL 
# DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, 
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING 
# OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""
import numpy as np
from types import SimpleNamespace


import torch
from torch.backends import cudnn
import torchvision.transforms as transforms


import os
import sys
sys.path.append(
    os.path.join(os.path.dirname(__file__), 'NVIDIA')
)
import network
from optimizer import restore_snapshot
from datasets import kitti
from config import assert_and_infer_cfg

sys.path.append(os.path.dirname(__file__))
from kitti_labels import cityscapes2kitti


class ImagePredictor:

    def __init__(self, snapshot, arch='network.deepv3.DeepWV3Plus'):
        self._process_args(snapshot, arch)
        self._build_net()
        self._build_img_transform()

    def _process_args(self, snapshot, arch):
        args = SimpleNamespace()
        args.snapshot = snapshot
        args.arch = arch
        assert_and_infer_cfg(args, train_mode=False)
        self.args = args
        cudnn.benchmark = False
        torch.cuda.empty_cache()


    def _build_net(self):
        self.args.dataset_cls = kitti
        net = network.get_net(self.args, criterion=None)
        net = torch.nn.DataParallel(net).cuda()
        print('Net built')
        net, _ = restore_snapshot(net, optimizer=None, snapshot=self.args.snapshot, restore_optimizer_bool=False)
        net.eval()
        print('Net restored')
        self.net = net
    
    def _build_img_transform(self):
        mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.img_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*mean_std)])
    
    def predict(self, image):
        img_tensor = self.img_transform(image)

        with torch.no_grad():
            img = img_tensor.unsqueeze(0).cuda()
            pred = self.net(img)
        
        pred = pred.cpu().numpy().squeeze()
        pred = np.argmax(pred, axis=0)

        return ImagePredictor.city2kitti_translate(pred)

    @staticmethod
    def city2kitti_translate(labels):
        return np.array([
            [ cityscapes2kitti[i] for i in row ] for row in labels
        ])
        

if __name__ == '__main__':

    if len(sys.argv) < 3:
        raise("Missing image or the pretrained model")
    import cv2

    snapshot = sys.argv[1]
    img = sys.argv[2]
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)

    predictor = ImagePredictor(snapshot)
    pred = predictor.predict(img)
    print(pred)

    if not len(sys.argv) > 3:
        exit(0)
    
    from kitti_labels import kitti_colors
    output = sys.argv[3]
    colored_img = np.array([
        [kitti_colors[l] for l in row] for row in pred
    ])
    cv2.imwrite(output, colored_img)
