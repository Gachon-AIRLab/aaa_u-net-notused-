import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
from gil_eval import *

from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.dataset import BasicDataset


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='MODEL.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=False)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=0.5)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))



def main(mode):
    #set cuda
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    #make model and load state
    model = UNet(n_channels=1, n_classes=1, bilinear=True)
    model.to(device)
    model.load_state_dict(torch.load('./checkpoints/u-net_1.pth'))


    #one image test
    if 'test' in mode:
        #link path
        check_dir = '../AAAGilDatasetPos/'
        subject = '05390853_20200821'
        img_idx = '35139583_20200821_0119'

        #in_files = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/raw/' + subject + '_%04d.png'%img_idx
        #mask_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/mask/' + subject + '_%04d.png'%img_idx
        
        in_files = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/raw/35139583_20200821_0119.png'
        mask_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/mask/35139583_20200821_0119.png'
        

        #open image
        img = Image.open(in_files)
        mask_gt = Image.open(mask_name)

        #get output (predict)
        prediction = predict_img(net=model, full_img=img,
                           scale_factor=1, out_threshold=0., device=device)

        
        mask = mask_to_image(prediction)
        #im = Image.fromarray((img * 255).astype(np.uint8))
        
        #mask.show()
        #mask_gt.show()


        img_in = np.array(img)
        img_mask = np.array(mask)
        img_mask_gt = np.array(mask_gt)


        cv2.imshow('input', img_in)
        cv2.imshow('mask result', img_mask)
        cv2.imshow('mask gt', img_mask_gt)
        cv2.waitKey(0)

        # segment evaluation
        overlap, jaccard, dice, fn, fp = eval_segmentation(img_mask, img_mask_gt)
        print('[segmentation evaluation] overlab:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f'%(overlap, jaccard, dice, fn, fp))


        #img_result = np.concatenate([img_mask, img_mask_gt], axis=1)

        #img_m = img_mask.unsqueeze(0)

        # img_overlap = img_gt.copy()
        # img_overlap[0, :] = 0
        # img_overlap[1 , :] = img_mask
        # cv2.imshow('result', img_result)
        # cv2.imshow('overlap', img_overlap)
        # cv2.waitKey(0)


    if 'eval' in mode:
        dataset_test = torch.load('./checkpoints/dataset_test_4.pth')

        num_test = len(dataset_test.indices)
        mat_eval = np.zeros((num_test, 5), np.float32)


        for i in range(num_test):
            img_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/raw/' + dataset_test.dataset.imgs[dataset_test.indices[i]]
            mask_name = '/media/suhyun/data/dataset/AAA/AAAGilDatasetPos/mask/' + dataset_test.dataset.imgs[dataset_test.indices[i]]

            #open image
            img = Image.open(img_name)
            mask_gt = Image.open(mask_name)

            #get output (predict)
            prediction = predict_img(net=model, full_img=img,
                           scale_factor=1, out_threshold=0.35, device=device)

        
            mask = mask_to_image(prediction)

            img_in = np.array(img)
            img_mask = np.array(mask)
            img_mask[img_mask > 50] = 255
            img_mask_gt = np.array(mask_gt)




            # segment evaluation
            overlap, jaccard, dice, fn, fp = eval_segmentation(img_mask, img_mask_gt)

            #print(dataset_test.dataset.imgs[dataset_test.indices[i]])

            print('[segmentation evaluation] overlab:%.4f jaccard:%.4f dice:%.4f fn:%.4f fp:%.4f' % (
            overlap, jaccard, dice, fn, fp))

            cv2.imshow('input', img_in)
            cv2.imshow('mask result', img_mask)
            cv2.imshow('mask gt', img_mask_gt)
            cv2.waitKey(0)



            mat_eval[i, :] = [overlap, jaccard, dice, fn, fp]

        print(mat_eval.mean(axis=0))


if __name__ == '__main__':
    main('eval')

