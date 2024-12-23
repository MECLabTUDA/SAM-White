#!/usr/bin/env python

"""
Example script to segment using our code base
"""

import SimpleITK as sitk
from model.utils import *
import os, argparse, shutil
from model.sam_networks import *
import dataloading.utils as d_utils

def segment():
    args = get_args()
    return _segment(args)

def get_args():
    # parse commandline args
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True, help='out path')
    parser.add_argument('--npz', required=True, help='pre-processed img npz file with embeddings and other stuff')
    parser.add_argument('--model', required=True, help='pytorch model for nonlinear registration')
    parser.add_argument('-g', '--gpu', help='GPU number(s) - if not supplied, CPU is used')
    parser.add_argument('--store_npz', action='store_true', help='Set this if the input should be copied as well --> takes a lot of space..')
    parser.add_argument('--use_bbox', action='store_true', default=False,
                        help='Set this if the bbox should be used in SAM.')
    parser.add_argument('--jitter',  type=float, default=0.0,
                        help='Specify the amount of jitter for the bbox, i.e. how much should it be enlarged.')
    parser.add_argument('--neg_samples', action='store_true', default=False,
                        help='Set this if negative samples points should be used as well.')
    parser.add_argument('--freeze_sam_body', action='store_true', default=True,
                        help='Set this if SAM body should be frozen during training.')
    parser.add_argument('--freeze_sam_head', action='store_true', default=False,
                        help='Set this if SAM head should be frozen during training.')
    parser.add_argument('--use_only_centroid_of_gt', action='store_true', default=False,
                        help='Set this if only the centroid sample of the GT should be used (independent of bbox). This overwrites nr_samples but not neg_samples.')
    parser.add_argument('--use_only_center_of_bbox', action='store_true', default=False,
                        help='Set this if only the center point of the bounding box should be used. This overwrites nr_samples but not neg_samples.')
    parser.add_argument('--use_quarter_four_points', action='store_true', default=False,
                        help='Set this if only the GT should be split in 4 and for every quarter one random sample should be used (independent of bbox). This overwrites nr_samples but not neg_samples.')
    
    args = parser.parse_args()
    return args

def _segment(args):
    # device handling
    device = 'cuda'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # load and set up model
    model = SAM.load(args.model, device)
    model.to(device)
    model.eval()

    # load input image (in this case we already pre-processed it using one of our scripts)
    imgs, segs, _, _, _, _, _, _ = d_utils.load_npzfile(args.npz, model.nr_samples, jitter=args.jitter, use_only_centroid_of_gt=args.use_only_centroid_of_gt, use_only_center_of_bbox=args.use_only_center_of_bbox, use_quarter_four_points=args.use_quarter_four_points)

    # Extract meta information on model
    epoch = int(args.model.split(os.sep)[-1][:-3])
    model_dir = os.path.join(os.sep, *args.model.split(os.sep)[:-1])
    
    # Do validation for one sample based on args.npz
    _, val_res, y_pred_ = validate(model, [args.npz], epoch, store_samples=True, out_=os.path.join(model_dir, "inference"), use_neg_samples=args.neg_samples, use_bbox=args.use_bbox, jitter=args.jitter, use_only_centroid_of_gt=args.use_only_centroid_of_gt, use_only_center_of_bbox=args.use_only_center_of_bbox, use_quarter_four_points=args.use_quarter_four_points)

    # Store npz finput file
    if args.store_npz:
        shutil.copy(args.npz, os.path.join(args.out, 'input.npz'))

    # Store image and GT segmentation
    sitk.WriteImage(sitk.GetImageFromArray(imgs[..., 0]), os.path.join(args.out, 'img.nii.gz'))
    sitk.WriteImage(sitk.GetImageFromArray(segs.squeeze()), os.path.join(args.out, 'seg_gt.nii.gz'))

    # Store predicted segmentation
    sitk.WriteImage(sitk.GetImageFromArray(y_pred_), os.path.join(args.out, 'pred_seg.nii.gz'))

    return val_res.loc[:, 'Dice'].mean(), val_res.loc[:, 'IoU'].mean()

# -- Main function for setup execution -- #
def main():
    segment()

if __name__ == "__main__":
    segment()