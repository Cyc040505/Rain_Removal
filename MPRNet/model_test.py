import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import Utils

from Dataset import get_test_data
from MPRNet import MPRNet
from skimage import img_as_ubyte
from pdb import set_trace as stx

parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='../dataset/test', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='../test_result/baseline', type=str, help='Directory for results')
parser.add_argument('--weights', default='../model/baseline.pth', type=str, help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

    model_restoration = MPRNet()

    Utils.load_checkpoint(model_restoration, args.weights)
    print("===>Testing using weights: ", args.weights)
    model_restoration.cuda()
    model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()

    datasets = ['Rain100L', 'Rain100H', 'Test100', 'Test1200', 'Test2800']

    for dataset in datasets:
        rgb_dir_test = os.path.join(args.input_dir, dataset, 'rain')
        test_dataset = get_test_data(rgb_dir_test, img_options={})
        test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4, drop_last=False,
                                 pin_memory=True)

        result_dir = os.path.join(args.result_dir, dataset)
        Utils.mkdir(result_dir)

        with torch.no_grad():
            for ii, data_test in enumerate(tqdm(test_loader), 0):
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()

                input_ = data_test[0].cuda()
                filenames = data_test[1]

                restored = model_restoration(input_)
                restored = torch.clamp(restored[0], 0, 1)

                restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

                for batch in range(len(restored)):
                    restored_img = img_as_ubyte(restored[batch])
                    Utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), restored_img)