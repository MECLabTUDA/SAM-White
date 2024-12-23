#!/usr/bin/env python

"""
Example script to train a SAM model with Adapter.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under the
License.
"""

import numpy as np
from model.utils import *
import itertools, monai
import torch, time, os, argparse
from model.sam_networks import *
from dataloading.generators import npz_generator

# Set seeds for numpy, random and pytorch
set_all_seeds(3299)
torch.set_printoptions(profile="full")


def train():
    args = get_args()
    _train(args)


def get_args():
    # parse the commandline
    parser = argparse.ArgumentParser()

    # data organization parameters
    parser.add_argument(
        "--train-dir",
        required=True,
        help="path to folder with pre-processed files (.npz)",
    )
    parser.add_argument("--model-dir", default="", help="model output directory.")
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Specify the amount of samples that should be extracted.",
    )
    parser.add_argument(
        "--use_bbox",
        action="store_true",
        default=False,
        help="Set this if the bbox should be used in SAM.",
    )
    parser.add_argument(
        "--jitter",
        type=float,
        default=0.0,
        help="Specify the amount of jitter for the bbox, i.e. how much should it be enlarged.",
    )
    parser.add_argument(
        "--neg_samples",
        action="store_true",
        default=False,
        help="Set this if negative samples points should be used as well.",
    )
    parser.add_argument(
        "--freeze_sam_body",
        action="store_true",
        default=True,
        help="Set this if SAM body should be frozen during training.",
    )
    parser.add_argument(
        "--freeze_sam_head",
        action="store_true",
        default=False,
        help="Set this if SAM head should be frozen during training.",
    )
    parser.add_argument(
        "--use_only_centroid_of_gt",
        action="store_true",
        default=False,
        help="Set this if only the centroid sample of the GT should be used (independent of bbox). This overwrites nr_samples but not neg_samples.",
    )
    parser.add_argument(
        "--use_only_center_of_bbox",
        action="store_true",
        default=False,
        help="Set this if only the center point of the bounding box should be used. This overwrites nr_samples but not neg_samples.",
    )
    parser.add_argument(
        "--use_quarter_four_points",
        action="store_true",
        default=False,
        help="Set this if only the GT should be split in 4 and for every quarter one random sample should be used (independent of bbox). This overwrites nr_samples but not neg_samples.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=25,
        help="Specify after how many epochs the model state is saved and validation is performed.",
    )

    # training parameters
    parser.add_argument(
        "--gpu", default="0", help="GPU ID number(s), comma-separated (default: 0)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="batch size (default: 1)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="number of training epochs (default: 250)",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=250,
        help="frequency of model saves (default: 100)",
    )
    parser.add_argument("--load-model", help="optional model file to initialize with")
    parser.add_argument(
        "--initial-epoch", type=int, default=0, help="initial epoch number (default: 0)"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--cudnn-nondet",
        action="store_true",
        help="disable cudnn determinism - might slow down training",
    )
    parser.add_argument("--model_type", type=str, default="vit_h", help="model type")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/sam_vit_l_0b3195.pth",
        help="checkpoint",
    )

    args = parser.parse_args()
    return args


def _train(args):
    # Extract vol names
    train_files = [
        os.path.join(args.train_dir, x)
        for x in os.listdir(args.train_dir)
        if ".npz" in x
    ]

    # -- Group for tasks to get eval splits between tasks -- #
    train_lists = [
        list(v)
        for _, v in itertools.groupby(train_files, key=lambda x: x.split(os.sep)[-4])
    ]  # --> task names and IDs

    for x in train_lists:
        random.shuffle(x)

    # -- Split into train and val based on tasks using 80:20 split -- #
    train_files_train = [
        x[: int((len(x) + 1) * 0.80)] for x in train_lists
    ]  # Remaining 80% to training set
    train_files_val = [
        x[int((len(x) + 1) * 0.80) :] for x in train_lists
    ]  # Splits 20% data to test set

    # -- Join the list of lists -- #
    train_files, val_files = [
        item for sublist in train_files_train for item in sublist
    ], [item for sublist in train_files_val for item in sublist]

    # -- Extract the number of samples -- #
    nr_samples = args.samples

    # load and prepare training data
    generator = npz_generator(
        train_files,
        nr_samples=nr_samples,
        jitter=args.jitter,
        use_only_centroid_of_gt=args.use_only_centroid_of_gt,
        use_only_center_of_bbox=args.use_only_center_of_bbox,
        use_quarter_four_points=args.use_quarter_four_points,
        batch_size=args.batch_size,
    )

    # extract shape from sampled input
    inshape = next(generator)[0][0].shape  # (256, 256, 3)
    assert inshape == (256, 256, 3)

    # prepare model folder
    model_dir = args.model_dir
    os.makedirs(model_dir, exist_ok=True)

    # device handling
    gpus = args.gpu.split(",")
    nb_gpus = len(gpus)
    device = "cuda"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert (
        np.mod(args.batch_size, nb_gpus) == 0
    ), "Batch size (%d) should be a multiple of the nr of gpus (%d)" % (
        args.batch_size,
        nb_gpus,
    )

    # enabling cudnn determinism appears to speed up training by a lot
    torch.backends.cudnn.deterministic = not args.cudnn_nondet

    if args.load_model:
        # load initial model (if specified)
        model = SAM.load(args.load_model, device)
    else:
        # otherwise configure new model
        # model = SAM_UNet(inshape=inshape, nr_samples=nr_samples, neg_samples=args.neg_samples, bbox=args.use_bbox,
        # model_type=args.model_type, checkpoint=args.checkpoint, device=device)
        model = SAM(
            inshape=inshape,
            nr_samples=nr_samples,
            neg_samples=args.neg_samples,
            bbox=args.use_bbox,
            model_type=args.model_type,
            checkpoint=args.checkpoint,
            device=device,
        )

    if nb_gpus > 1:
        # use multiple GPUs via DataParallel
        model = torch.nn.DataParallel(model)
        model.save = model.module.save

    # prepare the model for training and send to device
    model.to(device)
    model.train()

    # set optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # prepare loss with Dice
    losses = [monai.losses.DiceCELoss(sigmoid=True, reduction="mean")]
    weights = [1]

    # training loops
    for epoch in range(args.initial_epoch, args.epochs):
        # save model checkpoint
        if epoch % args.save_steps == 0:
            model.save(os.path.join(model_dir, "%04d.pt" % epoch))
            validate(
                model,
                val_files,
                epoch,
                store_samples=True,
                out_=os.path.join(args.model_dir, "validation"),
                use_neg_samples=args.neg_samples,
                use_bbox=args.use_bbox,
                jitter=args.jitter,
                use_only_centroid_of_gt=args.use_only_centroid_of_gt,
                use_only_center_of_bbox=args.use_only_center_of_bbox,
                use_quarter_four_points=args.use_quarter_four_points,
            )

        epoch_loss = []
        epoch_total_loss = []
        epoch_step_time = []
        dice = []

        for _ in range(args.steps_per_epoch):

            step_start_time = time.time()

            # Generate inputs (and true outputs) and convert them to tensors
            (
                imgs,
                segs,
                embeds,
                orig_sizes,
                input_sizes,
                bboxs_gt_,
                samples_gt_,
                neg_samples_gt_,
            ) = next(generator)
            imgs = torch.from_numpy(imgs).to(device).float()  # (n, 256, 256, 3)
            segs = torch.from_numpy(segs).to(device).float()  # (n, 1, 256, 256)
            # print(segs.shape)
            embeds = torch.from_numpy(embeds).to(device).float()  # (n, 1, 256, 64, 64)

            loss = 0
            loss_list = []

            # Run inputs through the model
            y_pred_ = model(
                imgs,
                embeds,
                orig_sizes,
                input_sizes,
                samples_gt_,
                neg_samples_gt_,
                bboxs_gt_,
                train=True,
                freeze_sam_body=args.freeze_sam_body,
                freeze_sam_head=args.freeze_sam_head,
            )

            nb_classes = segs.size(1) + 1  # Don't forget the background)

            # Build true and pred lists for loss calculation with Dice
            y_true = [segs.long()]
            y_pred = [y_pred_]

            # Calculate total loss
            for n, loss_function in enumerate(losses):
                if weights[n] == 0:
                    continue
                curr_loss = loss_function(y_pred[n], y_true[n]) * weights[n]
                loss_list.append(curr_loss.item())
                loss += curr_loss

            epoch_loss.append(loss_list)
            epoch_total_loss.append(loss.item())

            # Backpropagate and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate the Dice and print it to console --> maybe a log file as well?
            y_pred_ = torch.sigmoid(y_pred_)
            y_pred_ = y_pred_.detach().cpu().numpy().squeeze()
            y_pred_ = (y_pred_ > 0.5).astype(np.uint8)
            y_pred_ = get_one_hot(y_pred_, nb_classes)
            segs = get_one_hot(
                segs.detach().cpu().numpy().squeeze().astype(np.uint8), nb_classes
            )

            dice_ = DiceMetric(include_background=False, ignore_empty=False)(
                y_pred_, segs
            )
            dice.append(np.mean(dice_.numpy()) * 100)

            # Get compute time
            epoch_step_time.append(time.time() - step_start_time)

        # print epoch info
        epoch_info = "Epoch %d/%d" % (epoch + 1, args.epochs)
        time_step_info = "%.4f sec/step" % np.mean(epoch_step_time)
        time_epoch_info = "%.4f sec/epoch" % np.sum(epoch_step_time)
        losses_info = ", ".join(["%.4e" % f for f in np.mean(epoch_loss, axis=0)])
        loss_info = "loss: %.4e  (%s)" % (np.mean(epoch_total_loss), losses_info)
        dice_info = "dice: %.4f" % (np.mean(dice))
        print(
            " - ".join(
                (epoch_info, time_step_info, time_epoch_info, loss_info, dice_info)
            ),
            flush=True,
        )

    tot_params, train_params = get_nr_parameters(model)
    model_size = get_model_size(model)
    print(
        "Nr of parameter (total -- trainable): {} -- {}".format(
            tot_params, train_params
        )
    )
    print("Model size in MB: {:.4f}".format(model_size))

    # final model save and validate
    model.save(os.path.join(model_dir, "%04d.pt" % args.epochs))
    validate(
        model,
        val_files,
        args.epochs,
        store_samples=True,
        out_=os.path.join(args.model_dir, "validation"),
        use_neg_samples=args.neg_samples,
        use_bbox=args.use_bbox,
        jitter=args.jitter,
        use_only_centroid_of_gt=args.use_only_centroid_of_gt,
        use_only_center_of_bbox=args.use_only_center_of_bbox,
        use_quarter_four_points=args.use_quarter_four_points,
    )
    print("")


# -- Main function for setup execution -- #
def main():
    train()


if __name__ == "__main__":
    train()
