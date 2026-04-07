import torch
from torch.utils import data as data
from basicsr.data.data_util import read_img_seq
from basicsr.utils import scandir, get_root_logger
from os import path as osp
import glob

class VideoSRDataset(data.Dataset):
    """Video Super-Resolution dataset for training."""
    def __init__(self, opt):
        super(VideoSRDataset, self).__init__()
        self.opt = opt
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.num_frame = opt['num_frame']
        
        self.subfolders_lq = sorted(glob.glob(osp.join(self.lq_root, '*')))
        self.subfolders_gt = sorted(glob.glob(osp.join(self.gt_root, '*')))
        
        self.data_info = []
        for sub_lq, sub_gt in zip(self.subfolders_lq, self.subfolders_gt):
            img_paths_lq = sorted(list(scandir(sub_lq, full_path=True)))
            img_paths_gt = sorted(list(scandir(sub_gt, full_path=True)))
            max_idx = len(img_paths_lq)
            for i in range(0, max_idx - self.num_frame + 1):
                self.data_info.append({
                    'lq': img_paths_lq[i : i + self.num_frame],
                    'gt': img_paths_gt[i : i + self.num_frame]
                })

    def __getitem__(self, index):
        item = self.data_info[index]
        return {
            'lq': read_img_seq(item['lq']),
            'gt': read_img_seq(item['gt']),
            'lq_path': item['lq'][self.num_frame // 2]
        }

    def __len__(self):
        return len(self.data_info)