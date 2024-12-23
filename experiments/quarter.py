#!/usr/bin/env python
import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
from scripts import train
from config import *
from importlib import reload

# -- Set configurations manually -- #
nr_samples = 0
bbox = False  # False
jitter_bbox = 0.0  # Any value > 0 please, so the box will be enlarged
neg_samples = False  # False
save_steps = 50
nr_epochs = 75

# -- Set only one of those and if one is set, nr_samples and neg_samples won't be considered -- #
centroid_of_gt = False  # True
center_of_bbox = False  # True
quarter_four_points = True  # True

# -- Train only segmentation head of SAM -- #
freeze_sam_body = True
freeze_sam_head = False

# -- Train only full SAM --> NOTE: Not implemented yet -- #
# freeze_sam_body = False
# freeze_sam_head = False

continue_ = False
finished = False
continue_with_epoch = 0

# -- Train based on the configurations -- #
for tasks in train_on:
    trained_list = []
    for task in tasks:
        prev_mod_built = "_".join(trained_list)
        trained_list.append(task)
        built_ts = "_".join(trained_list)
        for model_type, checkpoint in zip(model_types, checkpoints):
            jitter_print = str(jitter_bbox).replace(".", "-")
            in_ = f"preprocessed_data/{mappings[task]}/{model_type}/Tr"
            out_folder = f"{RESULT_PATH}/sam_torch_{nr_epochs}_{built_ts}_ce_{nr_samples}_samples_use_neg_samples_{neg_samples}_sample_centroid_of_gt_{centroid_of_gt}_sample_center_of_bbox_{center_of_bbox}_samples_quarter_four_points_{quarter_four_points}_bbox_{bbox}_jitter_{jitter_print}_freeze_sam_body_{freeze_sam_body}_freeze_sam_head_{freeze_sam_head}"

            # -- Check if it is already trained or not -- #
            if os.path.exists(out_folder):
                # -- Started training on, so restore if more than one checkpoint -- #
                chks = [x for x in os.listdir(out_folder) if ".pt" in x]
                if len(chks) <= 1:  # Only 0000.pt in the list
                    if len(trained_list) > 1:  # <-- We still need load_model here
                        prev_model = out_folder.replace(built_ts, prev_mod_built)
                        continue_, finished, continue_with_epoch = True, True, 0
                        load_model = os.path.join(
                            prev_model, "%04d.pt" % nr_epochs
                        )  # <-- Should exist!
                    else:
                        continue_, finished, continue_with_epoch = False, False, 0
                else:
                    chks.sort()
                    chkp = chks[-1]
                    if str(nr_epochs) in chkp:
                        continue_, finished, continue_with_epoch = False, False, 0
                        continue  # <-- Finished with training for this task
                    continue_, finished, continue_with_epoch = (
                        True,
                        False,
                        int(chkp.split(".pt")[0][1:]),
                    )
                    load_model = os.path.join(
                        out_folder, "%04d.pt" % continue_with_epoch
                    )

            elif len(trained_list) > 1:  # <-- We still need load_model here
                prev_model = out_folder.replace(built_ts, prev_mod_built)
                continue_, finished, continue_with_epoch = True, True, 0
                load_model = os.path.join(
                    prev_model, "%04d.pt" % nr_epochs
                )  # <-- Should exist!

            # -- Build up arguments -- #
            args = [sys.argv[0], "--train-dir"]
            args += [in_]
            args += ["--model-dir", out_folder]
            if continue_:
                args += ["--load-model", load_model]
                if not finished:
                    args += ["--initial-epoch", str(continue_with_epoch)]
            args += ["--gpu", str(device)]
            args += ["--epochs", str(nr_epochs)]
            args += ["--save_steps", str(save_steps)]
            args += ["--model_type", model_type]
            args += ["--checkpoint", checkpoint]
            args += ["--samples", str(nr_samples)]
            if bbox:
                args += ["--use_bbox"]
                args += ["--jitter", str(jitter_bbox)]
            if neg_samples:
                args += ["--neg_samples"]
            if freeze_sam_body:
                args += ["--freeze_sam_body"]
            if freeze_sam_head:
                args += ["--freeze_sam_head"]
            if centroid_of_gt:
                args += ["--use_only_centroid_of_gt"]
            if center_of_bbox:
                args += ["--use_only_center_of_bbox"]
            if quarter_four_points:
                args += ["--use_quarter_four_points"]

            # -- Train -- #
            sys.argv = args

            train = reload(train)
            train.train()
