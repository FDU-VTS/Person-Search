import torch
from models.faceboxes import FaceBoxes
from models.reid import Baseline
from layers.functions.prior_box import PriorBox
from data import cfg
from utils.nms_wrapper import nms
from utils.box_utils import decode
import numpy as np
import torchvision.transforms as T
import time
import pymysql
import logging


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    # print(unused_pretrained_keys)
    missing_keys = model_keys - ckpt_keys
    logging.warning('Missing keys:{}'.format(len(missing_keys)))
    logging.warning('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    logging.warning('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    logging.warning('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def compare_vector(f1, f2):
    return torch.cosine_similarity(f1, f2)

def load_model(model, pretrained_path, load_to_cpu):
    logging.info('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

class Detect(object):
    def __init__(self, path, device):
        self.net = FaceBoxes(phase='test', size=None, num_classes=2).to(device)
        self.net = load_model(self.net, path, False)
        self.net.eval()
        self.device = device

    def get_bbox(self, img_raw):
        img = torch.FloatTensor(img_raw).to(self.device)
        im_height, im_width, _ = img.size()
        scale = torch.FloatTensor([im_width, im_height, im_width, im_height]).to(self.device)
        img -= torch.FloatTensor((104, 117, 123)).to(self.device)
        img = img.permute(2, 0, 1).unsqueeze(0)

        loc, conf = self.net(img)  # forward pass

        priorbox = PriorBox(cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > 0.05)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        # keep = py_cpu_nms(dets, args.nms_threshold)
        keep = nms(dets, 0.3, force_cpu=False)
        dets = dets[keep, :]

        # keep top-K faster NMS
        dets = dets[:750, :]
        bboxes = []
        for b in dets:
            if b[4] < 0.65:
                continue
            b = list(map(int, b))

            bboxes.append((b[0], b[1], b[2], b[3]))

        return bboxes


class Reid(object):
    def __init__(self, path, device):
        self.reid_model = Baseline(
            'resnet50',
            # 2494,
            1453,
            1,
            False,
            "ratio",
            (False, False, False, False),
            pretrain=False,
            model_path='./weights/model_best.pth').to(device)
        self.reid_model = load_model(self.reid_model, path, False)
        self.device = device
        self.reid_model.eval()
        self.transform = T.Compose([
            T.Resize([256, 128]),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def generate_feature(self, img):
        # print(self.transform(img).size())
        # print(np.asarray(img).shape)
        feature, temp = self.reid_model(self.transform(img).unsqueeze(0).to(self.device))

        # f_norm = torch.norm(feature, p=2, dim=1, keepdim=True)

        # feature = feature.div(f_norm.expand_as(feature))
        return feature, temp


if __name__ == "__main__":
    conn = pymysql.connect(user='root', password='root', database='aswbt', charset='utf8')
    cursor = conn.cursor()
    query = ('select id from bank')
    cursor.execute(query)
    for (id) in cursor:
        print(id)
    cursor.close()
    conn.close()
