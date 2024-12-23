import sys, os
import numpy as np
from tqdm import tqdm
from scripts import segment
from importlib import reload

# -- Set configurations manually -- #
built_ts = '700' # '900'
device = 0
nr_samples = 0
bbox = True # False
jitter_bbox = 0.2 # Any value > 0 please, so the box will be enlarged
jitter_print = str(jitter_bbox).replace('.', '-')
neg_samples = True # False
save_steps = 50
nr_epochs = 75

# -- Set only one of those and if one is set, nr_samples and neg_samples won't be considered -- #
centroid_of_gt = False# True
center_of_bbox = False # True
quarter_four_points = False # True

# -- Train only segmentation head of SAM -- #
freeze_sam_body = True
freeze_sam_head = False

# -- Train only full SAM --> NOTE: Not implemented yet -- #
# freeze_sam_body = False
# freeze_sam_head = False

ins = [
        # BCSS
        # '/media/aranem_locale/AR_subs_exps/SAM_white/preprocessed_data/Task900_BCSS/vit_b/Ts'

        # BraTS
        '/media/aranem_locale/AR_subs_exps1/SAM_white/preprocessed_data/Task700_BraTS2020/vit_b/Ts'
       ]

models = [
            f'/media/aranem_locale/AR_subs_exps1/SAM_white/trained_models/sam_torch_{nr_epochs}_{built_ts}_ce_{nr_samples}_samples_use_neg_samples_{neg_samples}_sample_centroid_of_gt_{centroid_of_gt}_sample_center_of_bbox_{center_of_bbox}_samples_quarter_four_points_{quarter_four_points}_bbox_{bbox}_jitter_{jitter_print}_freeze_sam_body_{freeze_sam_body}_freeze_sam_head_{freeze_sam_head}/0075.pt'
        #   '/media/aranem_locale/AR_subs_exps/SAM_white/trained_models/sam_torch_75_700_ce_0_samples_use_neg_samples_False_sample_centroid_of_gt_False_sample_center_of_bbox_False_samples_quarter_four_points_False_bbox_True_jitter_0-0_freeze_sam_body_True_freeze_sam_head_False/0075.pt'
          ]

# Extract predictions
for model in models:
    epoch = int(model.split(os.sep)[-1][:-3])
    for inp in ins:
        res_Dice, res_IoU = [], []
        print(f"Creating predictions with {model.split(os.sep)[-2]} for {inp.split(os.sep)[-3]}:")
        out_ = os.path.join(os.path.sep, *model.split(os.path.sep)[:-1], 'inference', "Epoch_"+str(epoch), inp.split(os.sep)[-3])
        cases = [x[:-4] for x in os.listdir(inp) if '._' not in x and '.json' not in x and 'DS_Store' not in x and '.npz' in x]
        cases = np.unique(cases)
        # for case in cases:
        for case in tqdm(cases):
            npz = os.path.join(inp, case+'.npz')
            out = os.path.join(out_, case)
            os.makedirs(out, exist_ok=True)

            # -- Build up arguments and do registration -- #
            args = [sys.argv[0], '--model']
            args += [model, '--npz']
            args += [npz]
            args += ['--out', out]
            # args += ['--samples', str(nr_samples)]
            if bbox:
                args += ['--use_bbox']
                args += ['--jitter', str(jitter_bbox)]
            if neg_samples:
                args += ['--neg_samples']
            if freeze_sam_body:
                args += ['--freeze_sam_body']
            if freeze_sam_head:
                args += ['--freeze_sam_head']
            if centroid_of_gt:
                args += ['--use_only_centroid_of_gt']
            if center_of_bbox:
                args += ['--use_only_center_of_bbox']
            if quarter_four_points:
                args += ['--use_quarter_four_points']
            sys.argv = args

            segment = reload(segment)   # So the log files can be updated as well
            Dice, IoU = segment.segment()
            res_Dice.append(Dice)
            res_IoU.append(IoU)
        print(f"Performance of model {model.split(os.sep)[-2]} for {inp.split(os.sep)[-2]} (Dice -- IoU): {np.mean(res_Dice):.2f}% -- {np.mean(res_IoU):.2f}%")
