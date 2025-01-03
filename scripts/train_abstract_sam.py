#!/usr/bin/env python

import sys, os
from scripts import train
from importlib import reload
# import unet

# -- Set configurations manually -- #
device = 0
nr_samples = 0
bbox = True # False
jitter_bbox = 0.0 # Any value > 0 please, so the box will be enlarged
neg_samples = False # False
save_steps = 50
nr_epochs = 75

# -- Set only one of those and if one is set, nr_samples and neg_samples won't be considered -- #
centroid_of_gt = False# True
center_of_bbox = True # True
quarter_four_points = False # True

# -- Train only segmentation head of SAM -- #
freeze_sam_body = True
freeze_sam_head = False

# -- Train only full SAM --> NOTE: Not implemented yet -- #
# freeze_sam_body = False
# freeze_sam_head = False

mappings = {
            '700': 'Task700_BraTS2020',
            '808': 'Task808_mHeartA',
            '900': 'Task900_BCSS',            
           }
train_on = [
            ['700'],
            #['900'],
           ]

continue_ = False
finished = False
continue_with_epoch = 0

# model_types = ['vit_b', 'vit_h', 'vit_l']
# checkpoints = ['/media/aranem_locale/AR_subs_exps/SAM_white/checkpoints/sam_vit_b_01ec64.pth',
#                '/media/aranem_locale/AR_subs_exps/SAM_white/checkpoints/sam_vit_h_4b8939.pth',
#                '/media/aranem_locale/AR_subs_exps/SAM_white/checkpoints/sam_vit_l_0b3195.pth']

model_types = ['vit_b']
checkpoints = ['/media/aranem_locale/AR_subs_exps/SAM_white/checkpoints/sam_vit_b_01ec64.pth']

# -- Train based on the configurations -- #
for tasks in train_on:
    trained_list = []
    for task in tasks:
        prev_mod_built = '_'.join(trained_list)
        trained_list.append(task)
        built_ts = '_'.join(trained_list)
        for model_type, checkpoint in zip(model_types,checkpoints):
            jitter_print = str(jitter_bbox).replace('.', '-')
            in_ = f'/media/aranem_locale/AR_subs_exps/SAM_white/preprocessed_data/{mappings[task]}/{model_type}/Tr'
            # in_ = f'/home/nbabende/PythonProjects/SAM_pathology/radio/{mappings[task]}/{model_type}/Tr'
            out_folder = f'/media/aranem_locale/AR_subs_exps/SAM_white/trained_models/sam_torch_{nr_epochs}_{built_ts}_ce_{nr_samples}_samples_use_neg_samples_{neg_samples}_sample_centroid_of_gt_{centroid_of_gt}_sample_center_of_bbox_{center_of_bbox}_samples_quarter_four_points_{quarter_four_points}_bbox_{bbox}_jitter_{jitter_print}_freeze_sam_body_{freeze_sam_body}_freeze_sam_head_{freeze_sam_head}'
            # out_folder = f'/home/aranem_locale/Desktop/mnts/local/scratch/aranem/SAM_white/trained_models/sam_torch_{nr_epochs}_{built_ts}_ce_{nr_samples}_samples_use_neg_samples_{neg_samples}_sample_centroid_of_gt_{centroid_of_gt}_sample_center_of_bbox_{center_of_bbox}_samples_quarter_four_points_{quarter_four_points}_bbox_{bbox}_jitter_{jitter_print}_freeze_sam_body_{freeze_sam_body}_freeze_sam_head_{freeze_sam_head}'

            # -- Check if it is already trained or not -- #
            if os.path.exists(out_folder):
                # -- Started training on, so restore if more than one checkpoint -- #
                chks = [x for x in os.listdir(out_folder) if '.pt' in x]
                if len(chks) <= 1:  # Only 0000.pt in the list
                    if len(trained_list) > 1: # <-- We still need load_model here
                        prev_model = out_folder.replace(built_ts, prev_mod_built)
                        continue_, finished, continue_with_epoch = True, True, 0
                        load_model = os.path.join(prev_model, '%04d.pt' % nr_epochs)    # <-- Should exist!
                    else:
                        continue_, finished, continue_with_epoch = False, False, 0
                else:
                    chks.sort()
                    chkp = chks[-1]
                    if str(nr_epochs) in chkp:
                        continue_, finished, continue_with_epoch = False, False, 0
                        continue    # <-- Finished with training for this task
                    continue_, finished, continue_with_epoch = True, False, int(chkp.split('.pt')[0][1:])
                    load_model = os.path.join(out_folder, '%04d.pt' % continue_with_epoch)

            elif len(trained_list) > 1: # <-- We still need load_model here
                prev_model = out_folder.replace(built_ts, prev_mod_built)
                continue_, finished, continue_with_epoch = True, True, 0
                load_model = os.path.join(prev_model, '%04d.pt' % nr_epochs)    # <-- Should exist!

            # -- Build up arguments -- #
            args = [sys.argv[0], '--train-dir']
            args += [in_]
            args += ['--model-dir', out_folder]
            if continue_:
                args += ['--load-model', load_model]
                if not finished:
                    args += ['--initial-epoch', str(continue_with_epoch)]
            args += ['--gpu', str(device)]
            args += ['--epochs', str(nr_epochs)]
            args += ['--save_steps', str(save_steps)]
            args += ['--model_type', model_type]
            args += ['--checkpoint', checkpoint]
            args += ['--samples', str(nr_samples)]
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

            # -- Train -- #
            sys.argv = args

            train = reload(train)
            train.train()