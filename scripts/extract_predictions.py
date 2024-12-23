import sys, os

sys.path.append(os.path.dirname(__file__) + "/..")
import numpy as np
from tqdm import tqdm
from scripts import segment
from config import *
from importlib import reload
from pathlib import Path

# -- Set configurations manually -- #
# built_ts = train_on[0]
# nr_samples = 100
# bbox = False  # False
# jitter_bbox = 0.0  # Any value > 0 please, so the box will be enlarged
# jitter_print = str(jitter_bbox).replace(".", "-")
# neg_samples = False  # False
# save_steps = 50
# nr_epochs = 75

# -- Set only one of those and if one is set, nr_samples and neg_samples won't be considered -- #
# centroid_of_gt = False  # True
# center_of_bbox = False  # True
# quarter_four_points = False  # True

# -- Train only segmentation head of SAM -- #
# freeze_sam_body = True
# freeze_sam_head = False

# -- Train only full SAM --> NOTE: Not implemented yet -- #
# freeze_sam_body = False
# freeze_sam_head = False

task = "Task300_KvasirSEG"

ins = [f"preprocessed_data/{task}/vit_b/Ts"]


def options_from_dirname(dirname):
    tokens = dirname.split("_")
    epoch = int(tokens[2])
    built_ts = tokens[3]
    nr_samples = int(tokens[5])

    neg_samples = tokens[10] == "True"
    centroid_of_gt = tokens[15] == "True"
    center_of_bbox = tokens[20] == "True"
    quarter_four_points = tokens[25] == "True"
    bbox = tokens[27] == "True"
    jitter_bbox = float(tokens[29].replace("-", "."))
    freeze_sam_body = tokens[33] == "True"
    freeze_sam_head = tokens[37] == "True"
    return (
        epoch,
        built_ts,
        nr_samples,
        neg_samples,
        centroid_of_gt,
        center_of_bbox,
        quarter_four_points,
        bbox,
        jitter_bbox,
        freeze_sam_body,
        freeze_sam_head,
    )


for model_path in Path(RESULT_PATH).glob("*"):

    # models = [
    #    f"{RESULT_PATH}/sam_torch_{nr_epochs}_{built_ts}_ce_{nr_samples}_samples_use_neg_samples_{neg_samples}_sample_centroid_of_gt_{centroid_of_gt}_sample_center_of_bbox_{center_of_bbox}_samples_quarter_four_points_{quarter_four_points}_bbox_{bbox}_jitter_{jitter_print}_freeze_sam_body_{freeze_sam_body}_freeze_sam_head_{freeze_sam_head}/0075.pt"
    # ]

    # Extract predictions
    print(model_path)
    dirname = model_path.name

    (
        epoch,
        built_ts,
        nr_samples,
        neg_samples,
        centroid_of_gt,
        center_of_bbox,
        quarter_four_points,
        bbox,
        jitter_bbox,
        freeze_sam_body,
        freeze_sam_head,
    ) = options_from_dirname(dirname)

    model = model_path / f"{epoch:04d}.pt"

    for inp in ins:
        res_Dice, res_IoU = [], []
        # print(
        #    f"Creating predictions with {model.split(os.sep)[-2]} for {inp.split(os.sep)[-3]}:"
        # )
        out_ = model_path / "inference" / f"Epoch_{epoch}" / inp.split(os.sep)[-3]
        cases = [
            x[:-4]
            for x in os.listdir(inp)
            if "._" not in x
            and ".json" not in x
            and "DS_Store" not in x
            and ".npz" in x
        ]
        cases = np.unique(cases)

        for case_ in tqdm(cases):
            npz = os.path.join(inp, case_ + ".npz")
            out = os.path.join(out_, case_)
            os.makedirs(out, exist_ok=True)

            # -- Build up arguments and do registration -- #
            args = [sys.argv[0], "--model"]
            args += [str(model), "--npz"]
            args += [npz]
            args += ["--out", out]
            # args += ['--samples', str(nr_samples)]
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
            sys.argv = args

            segment = reload(segment)  # So the log files can be updated as well
            Dice, IoU = segment.segment()
            res_Dice.append(Dice)
            res_IoU.append(IoU)
        print(
            f"Performance of model {model_path} for {inp.split(os.sep)[-2]} (Dice -- IoU): {np.mean(res_Dice):.2f}% -- {np.mean(res_IoU):.2f}%"
        )
