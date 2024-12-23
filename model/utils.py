import numpy as np
import pandas as pd
from skimage import io
import dataloading.utils as d_utils
import os, random, torch, copy, cv2#, warnings
from monai.metrics import DiceMetric, MeanIoU


def set_all_seeds(seed):
  random.seed(seed)
  # os.environ("PYTHONHASHSEED") = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def get_nr_parameters(model):
    r"""This function returns the number of parameters and trainable parameters of a network.
        Based on: https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model
    """
    # -- Extract and count nr of parameters -- #
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # -- Return the information -- #
    return total_params, trainable_params

def get_model_size(model):
    r"""This function return the size in MB of a model.
        Based on: https://discuss.pytorch.org/t/finding-model-size/130275
    """
    # -- Extract parameter and buffer sizes -- #
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    # -- Transform into MB -- #
    size_all_mb = (param_size + buffer_size) / 1024**2
    # -- Return the size -- #
    return size_all_mb

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    res = torch.from_numpy(res)
    return res.reshape(list(targets.shape)+[nb_classes]).permute(0, 3, 1, 2)

def get_mask_from_bbox_coord(coords, shape=(1, 256, 256)):
    r"""
        This function gets 4 coordinates and returns a numpy binary mask, where the area in the bounding box is segmented.
        :param coords: List of 4 coordinates in the form: [N, 4], each coordinate having the format [X1, X2, Y1, Y2]
        :param shape: The final shape of the mask to be returned
    """
    mask = np.zeros(shape, dtype=np.uint8)               # initialize mask
    for idx, coord in enumerate(coords):
        # Calculate the bbox area from coordinates before slicing
        # Xmin, Xmax, Ymin, Ymax = coord
        X1, X2, Y1, Y2 = coord
        # H, W = abs(Y1 - Y2), abs(X1 - X2)
        # mask[idx, int(Ymin):int(Ymax), int(Xmin):int(Xmax)] = 1  # fill with white pixels
        mask[idx, int(Y2):int(Y1), int(X2):int(X1)] = 1  # fill with white pixels, note: always Y2 > Y1 and X2 > X1
        # mask[idx, int(Y1):int(Y1 + H), int(X1):int(X1 + W)] = 1  # fill with white pixels
        # mask[idx, int(coord[2]):int(coord[3]), int(coord[0]):int(coord[1])] = 1  # fill with white pixels
    return mask

def extract_samples_from_tensor(samples_tensor, nr_samples):
    r"""
        Use this to create a list of 2D coordinates (samples) from a tensor resulting in (x, y) coordinate pairs.
    """
    samples_list = list()
    for i in range(nr_samples):
        samples_list.append(samples_tensor[i].detach().cpu().numpy().squeeze())
    return np.asarray(samples_list)


def plot_slice_with_samples_bboxs_png(slice, samples, neg_samples, bboxs, out, plot_neg=False, use_bbox=True):
    """
    This function takes a slice with sample and bounding box coordinates and stores the image with the
    samples and bbox as a png image using the specified out path.
        :param slice: A 2D RGB slice of an image.
        :param samples: List of samples, i.e. coordinates within the image.
        :param bbox: List of 4 coordinates, i.e. coordinates of a bounding box within the image --> [X1, X2, Y1, Y2].
        :param out: Path to where the slice with plotted samples and bbox should be stored.
    """
    # Build png image
    png = copy.deepcopy(slice)

    # Make dots for neg_samples red
    if plot_neg:
        for sample in neg_samples:
            try:
                png[int(sample[0]), int(sample[1])] = [255, 0, 0]
            except IndexError:
                pass
                #png[0, 0] = [255, 0, 0]
    
    # Build png image
    png = copy.deepcopy(slice)

    # Make dots for neg_samples red
    if plot_neg:
        for sample in neg_samples:
            try:
                cv2.circle(png, [int(sample[1]), int(sample[0])], 2, [255, 0, 0], 2)

            except IndexError:
                pass
                # png[0, 0] = [255, 0, 0]

    # Make dots for samples green
    for sample in samples:
        try:
            cv2.circle(png, [int(sample[1]), int(sample[0])], 2, [0, 255, 0], 2)
        except IndexError:
            pass
            # png[0, 0] = [0, 255, 0]

    # Make lines for bounding box green
    if use_bbox:  # <-- It is 0 if it is not predicted/present
        for bbox in bboxs:
            if len(bbox) != 0:
                start_point = (
                    int(bbox[0]),
                    int(bbox[2]),
                )  # represents the top left corner of rectangle (X1, Y1)
                end_point = (
                    int(bbox[1]),
                    int(bbox[3]),
                )  # represents the bottom right corner of rectangle (X2, Y2)
                cv2.rectangle(
                    png, start_point, end_point, color=(0, 0, 255), thickness=2
                )
    
    # Store the image
    io.imsave(out, png, check_contrast=False)


def validate(model, val_list, epoch, out_, store_samples=False, npz_=True, use_neg_samples=False, use_bbox=True, jitter=0.,use_only_centroid_of_gt=False, use_only_center_of_bbox=False, use_quarter_four_points=False):
    r"""
    Call this function in order to validate the model with data from the provided generator.
        :param model: The model architecture to use for predictions.
        :param val_list: List of paths to pre-processed validation cases (.npz).
        :param epoch: Integer of the current epoch we perform the validation in
        :param out_: Path to where the metrics (and samples) should be stored
        :param store_samples: Set this flag if png images should be generated showing the samples and bounding boxes
        :param npz_: Indicates if the list contains paths to npz files or not. In the latter the pre-processing will be performed.
    """
    # Put model in eval mode and initialize dictionaries
    model.eval()
    val_res = pd.DataFrame()
    os.makedirs(out_, exist_ok=True)
    device = next(model.parameters()).device

    for npz in val_list:
        # Extract the case name
        case = npz.split(os.sep)[-1].split('.npz')[0]
        task = npz.split(os.sep)[-4]
        # Load input image (in this case we already pre-processed it, i.e. extracted embeddings and fixed shape issues using one of our scripts)
        if npz_:
            imgs_, segs_, embeds, orig_sizes, input_sizes, bboxs_gt, samples_gt, neg_samples_gt = d_utils.load_npzfile(npz, model.nr_samples, jitter=jitter, use_only_centroid_of_gt=use_only_centroid_of_gt, use_only_center_of_bbox=use_only_center_of_bbox, use_quarter_four_points=use_quarter_four_points)
        else:
            # TODO: Add alternative if the image is nii.gz and not a npz file, i.e. not pre-processed
            pass
        
        imgs = torch.from_numpy(imgs_).to(device).float()
        segs = torch.from_numpy(segs_).to(device).float()
        embeds = torch.from_numpy(embeds).to(device).float()

        # Predict
        _y_pred_ = model(imgs, embeds, orig_sizes, input_sizes, samples_gt, neg_samples_gt, bboxs_gt, train=False)    # Only use the ones for plotting here

        # Prepare predictions and GT
        nb_classes = segs.size(1) + 1 # Don't forget the background
        y_pred_ = get_one_hot(_y_pred_, nb_classes)
        segs = get_one_hot(segs.detach().cpu().numpy().squeeze().astype(np.uint8), nb_classes)

        # Calculate metrics
        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")
        #     HD = HausdorffDistanceMetric(include_background=False)(y_pred_, segs)
        Dice = DiceMetric(include_background=False, ignore_empty=False)(y_pred_, segs)
        IoU = MeanIoU(include_background=False, ignore_empty=False)(y_pred_, segs)
        # MSE_bbox = MSEMetric()(bbox, torch.from_numpy(bboxs_gt))
        # MSE_samples = MSEMetric()(samples, torch.from_numpy(samples_gt))

        # Append to dataframe
        val_res = pd.concat([val_res,
                                  pd.DataFrame.from_records([{'Epoch': str(epoch), 'Task': task, 'ID': case, #'HD': np.mean(HD.numpy()),
                                                              'Dice': np.mean(Dice.numpy()*100),
                                                              'IoU': np.mean(IoU.numpy()*100),
                                                            #   'MSE (samples)': np.mean(MSE_samples.numpy()),
                                                            #   'MSE (bbox)': np.mean(MSE_bbox.numpy()),
                                                              }])
                            ], axis=0)
        
        # Store pngs with samples to show progress during training if desired
        if store_samples:
            # Store a png with bounding box and samples for every slice
            out = os.path.join(out_, "Epoch_"+str(epoch), task, case)
            os.makedirs(out, exist_ok=True)
            
            # Generate for every slice the pngs and store them under out/case_slice_ID.png
            for i, slice in enumerate(imgs_):
                # Plot the created GT labels and bounding box use this line below
                plot_slice_with_samples_bboxs_png(slice, samples_gt[i], neg_samples_gt[i],
                                                  bboxs_gt[i], os.path.join(out, "slice_"+str(i)+'.png'), plot_neg=use_neg_samples, use_bbox=use_bbox)

    # Store metrics csv
    if not os.path.isfile(os.path.join(out_, "validation_results.csv")):
        val_res.to_csv(os.path.join(out_, "validation_results.csv"), index=False, sep=',')  # <-- Includes the header
    else: # else it exists so append without writing the header
        val_res.to_csv(os.path.join(out_, "validation_results.csv"), index=False, sep=',', mode='a', header=False)  # <-- Omits the header 

    # Put model back into train mode and return the results
    model.train()
    return y_pred_, val_res, _y_pred_  # --> patient wise, not batch wise