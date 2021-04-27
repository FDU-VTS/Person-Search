import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import glob
import re
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


WIDER_CLASSES = ( '__background__', 'person')


class AnnotationTransform(object):

    """Transforms a VOC annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes

    Arguments:
        class_to_ind (dict, optional): dictionary lookup of classnames -> indexes
            (default: alphabetic indexing of VOC's 20 classes)
        keep_difficult (bool, optional): keep difficult instances or not
            (default: False)
        height (int): height
        width (int): width
    """

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(WIDER_CLASSES, range(len(WIDER_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):
        """
        Arguments:
            target (annotation) : the target annotation to be made usable
                will be an ET.Element
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class name]
        """
        res = np.empty((0, 5))
        for obj in target.iter('object'):
            difficult = int(obj.find('difficult').text) == 1
            if not self.keep_difficult and difficult:
                continue
            name = obj.find('name').text.lower().strip()
            if name != 'person':
                continue
            bbox = obj.find('bndbox')

            pts = ['xmin', 'ymin', 'xmax', 'ymax']
            bndbox = []
            for i, pt in enumerate(pts):
                cur_pt = int(bbox.find(pt).text)
                bndbox.append(cur_pt)
            label_idx = self.class_to_ind[name]
            bndbox.append(label_idx)
            res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]
        return res

class AnnotationTransformTXT(object):

    def __init__(self, class_to_ind=None, keep_difficult=True):
        self.class_to_ind = class_to_ind or dict(
            zip(WIDER_CLASSES, range(len(WIDER_CLASSES))))
        self.keep_difficult = keep_difficult

    def __call__(self, target):

        res = np.empty((0, 5))

        line_list = target.readlines()

        str_line = ''
        for line in line_list:
            if str(line).__contains__('Image filename'):
                str_line = line.strip().split('/')[2][0:-1]  # remove the end of "
                break

        for line in line_list:
            if str(line).__contains__('Objects with ground truth'):
                nums = re.findall(r'\d+', str(line))
                str_line = str_line + ' ' + str(nums[0])
                # print(str_line)
                break

        for index in range(1, int(nums[0]) + 1):
            for line in line_list:
                if str(line).__contains__("Bounding box for object " + str(index)):
                    bndbox = []
                    coordinate = re.findall(r'\d+', str(line))
                    str_line = str_line + ' ' + coordinate[1] + ' ' + coordinate[2] + ' ' + coordinate[3] + ' ' + \
                               coordinate[4]
                    bndbox.append(int(coordinate[1]))
                    bndbox.append(int(coordinate[2]))
                    bndbox.append(int(coordinate[3]))
                    bndbox.append(int(coordinate[4]))
                    label_idx = self.class_to_ind['person']
                    bndbox.append(label_idx)
                    res = np.vstack((res, bndbox))  # [xmin, ymin, xmax, ymax, label_ind]

        target.close()
        return res


class VOCDetection(data.Dataset):

    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to WIDER folder
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
    """

    def __init__(self, root, preproc=None, target_transform=None):
        self.root = root
        self.preproc = preproc
        self.target_transform = target_transform
        self._annopath = os.path.join(self.root, 'annotations', '%s')
        self._imgpath = os.path.join(self.root, 'images', '%s')
        self.ids = list()
        # with open(os.path.join(self.root, 'img_list.txt'), 'r') as f:
        #   self.ids = [tuple(line.split()) for line in f]
        self.ids = [(line, line.replace(".png", ".xml").replace('.jpg', '.xml').replace("images", "labels")) for line in
                    glob.glob(self.root + "/images/*.png") + glob.glob(self.root + "/images/*.jpg")]

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = ET.parse(img_id[1]).getroot()
        # target = open(img_id[1])
        img = cv2.imread(img_id[0], cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target

    def __len__(self):
        return len(self.ids)


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)
