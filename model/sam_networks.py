import numpy as np
import torch, copy
from dataloading.utils import isListEmpty
from segment_anything import sam_model_registry
from model.utils import extract_samples_from_tensor
from .modelio import LoadableModel, store_config_args
from segment_anything.utils.transforms import ResizeLongestSide

# from unet import UNet2D, UNet3D, UNet
# from diffusers import UNet2DModel


class SAM(LoadableModel):
    """Med_Sam network for medical imaging."""

    @store_config_args
    def __init__(
        self,
        inshape,
        nr_samples,
        neg_samples=False,
        bbox=False,
        model_type="vit_b",
        checkpoint="checkpoints/sam_vit_b_01ec64.pth",
        device="cpu",
    ):
        super(SAM, self).__init__()
        self.device = device
        self.inshape = inshape
        self.nr_samples = nr_samples
        self.use_bbox = bbox
        self.use_neg_samples = neg_samples

        # -- Initialize and load SAM -- #
        self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
        self.transform = ResizeLongestSide(self.sam_model.image_encoder.img_size)

    def forward(
        self,
        x,
        x_embed,
        orig_size,
        input_size,
        samples_c,
        neg_samples_c=None,
        bbox_c=None,
        train=True,
        freeze_sam_body=False,
        freeze_sam_head=False,
        **kwargs
    ):
        r"""
        Forward pass.
        param x: input image RGB Size([N, 256, 256, 3])
        param x_embed: input image embeding already done with SAM (preprocessing)
        param orig_size: Original size of input image
        param input_size: Size of transformed image
        param train: Set this flag to specify if the call is made during training or not, depending on that the segmentation will be returned differently
        param freeze_sam_body: Set this flag if the SAM body should be frozen
        param freeze_sam_head: Set this flag if the SAM segmentation head should be frozen
        param freeze_adapter: Set this flag if the Adapter should be frozen
        """
        if self.use_bbox:
            assert (
                bbox_c is not None
            ), "If you want to use bbox, then provide a bbox that is not None."

        segs = list()
        for i, _ in enumerate(x):
            # -- Build prompt -- #
            samples_, neg_samples_ = copy.deepcopy(samples_c[i]), copy.deepcopy(
                neg_samples_c[i]
            )  # <-- Otherwise in-place changes

            if self.use_neg_samples:
                if not isListEmpty(samples_):
                    samples_.extend(neg_samples_)
                else:
                    samples_ = copy.deepcopy(neg_samples_)

            samples_ = torch.from_numpy(np.asarray(samples_)).to(x.device).float()

            # Create samples into pairs
            samples_len = 0 if isListEmpty(samples_c[i]) else len(samples_c[i])
            neg_samples_len = (
                0 if isListEmpty(neg_samples_c[i]) else len(neg_samples_c[i])
            )

            if self.use_neg_samples:
                input_points = extract_samples_from_tensor(
                    samples_, samples_len + neg_samples_len
                )  # Size([nr_samples, 2])
            else:
                input_points = extract_samples_from_tensor(
                    samples_, samples_len
                )  # Size([nr_samples, 2])

            input_labels = np.array(
                [1] * samples_len
            )  # Corresponding label for the points --> since binary for now, always 1
            input_labels_neg = np.array(
                [0] * neg_samples_len
            )  # Corresponding label for the points --> since binary for now, always 0

            if self.use_neg_samples:
                # oin input_labels and input_labels_neg
                input_labels = np.concatenate((input_labels, input_labels_neg))

            if self.use_bbox:
                bboxs = np.asarray(bbox_c[i])  # Size([x, 4]) <-- x := nr of CCs

            with torch.set_grad_enabled(
                not freeze_sam_body
            ):  # If freeze_sam_body True then use torch.no_grad
                # -- Set the slice in SAM manually using the already extracted embedding -- #
                self.features = x_embed[
                    i
                ].clone()  # <-- Only if freeze_sam_body, otherwise calculate them new
                if freeze_sam_body:
                    # TODO: Extract self.features
                    pass

                self.original_size = orig_size[0]
                self.input_size = input_size[0]
                self.is_image_set = True

                # -- Do SAM predict -- #
                # -- Sampled points -- #
                if not isListEmpty(input_points):
                    point_coords = self.transform.apply_coords(
                        input_points, self.original_size
                    )
                    coords_torch = torch.as_tensor(
                        point_coords, dtype=torch.float, device=self.device
                    )
                    labels_torch = torch.as_tensor(
                        input_labels, dtype=torch.int, device=self.device
                    )
                    coords_torch, labels_torch = (
                        coords_torch[None, :, :],
                        labels_torch[None, :],
                    )

                if self.use_bbox:
                    # -- Bounding Box -- #
                    bboxs_new = []
                    for box in bboxs:
                        if len(box) != 0:
                            bboxs_new.append(
                                box
                            )  # <-- If even one box is [], then don't add it
                    if len(bboxs_new) != 0:
                        box_ = self.transform.apply_boxes(
                            np.asarray(bboxs_new), self.original_size
                        )
                        box_torch = torch.as_tensor(
                            box_, dtype=torch.float, device=self.device
                        ).unsqueeze(1)
                    else:  # <-- If all provided bboxs are [], then set bbox to None
                        box_torch = None

                # Embed prompts
                sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                    points=(
                        (coords_torch, labels_torch)
                        if not isListEmpty(input_points)
                        else None
                    ),
                    boxes=box_torch if self.use_bbox else None,
                    masks=None,
                )

            # Predict masks
            with torch.set_grad_enabled(
                not freeze_sam_head
            ):  # If freeze_sam_head True then use torch.no_grad
                low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                    image_embeddings=self.features,
                    image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )

            segs.append(low_res_masks[0])

        segs = torch.stack(segs, dim=0).float()  # Size([N, 1, 256, 256])

        if not train:
            segs = torch.sigmoid(segs)
            segs = segs.detach().cpu().numpy().squeeze()
            segs = (segs > 0.5).astype(np.uint8)

        return segs


class SAM_UNet(SAM):
    @store_config_args
    def __init__(
        self,
        inshape,
        nr_samples,
        neg_samples=False,
        bbox=False,
        model_type="vit_b",
        checkpoint="/home/aranem_locale/Desktop/SAM_CL/checkpoints/sam_vit_b_01ec64.pth",
        device="cpu",
    ):
        super(SAM_UNet, self).__init__(
            inshape, nr_samples, neg_samples, bbox, model_type, checkpoint, device
        )
        self.unet = UNet()

    def forward(
        self,
        x,
        x_embed,
        orig_size,
        input_size,
        samples_c,
        neg_samples_c=None,
        bbox_c=None,
        train=True,
        freeze_sam_body=False,
        freeze_sam_head=False,
        freeze_unet=False,
        **kwargs
    ):
        r"""
        Forward pass.
        """
        # Forward SAM with always train, as we don't need sigmoid for U-Net and SAM body is always frozen.
        # NOTE: If the seg head is trained in first stage and then turned frozen, this should be specified using freeze_sam_head!
        segs = super(SAM_UNet, self).forward(
            x,
            x_embed,
            orig_size,
            input_size,
            samples_c,
            neg_samples_c,
            bbox_c,
            True,
            False,
            freeze_sam_head,
            **kwargs
        )
        # Forward UNet
        if not freeze_unet:
            segs = self.unet(segs)

        # During inference or not train stages do sigmoid etc. now, after the UNet, not before it !
        if not train:
            segs = torch.sigmoid(segs)
            segs = segs.detach().cpu().numpy().squeeze()
            segs = (segs > 0.5).astype(np.uint8)

        return segs


class SAM_Path(LoadableModel):
    @store_config_args
    def __init__(self):
        super(SAM_Path, self).__init__()
        # @Niklas: TODO add the class initialization and all you need here

    def forward(self, x):
        # @Niklas: TODO add the forward pass here
        return x


class UNet(LoadableModel):
    @store_config_args
    def __init__(self):
        super(UNet, self).__init__()
        # @Niklas: TODO add the class initialization and all you need here
        # unet=unetclass
        # Initialize of the shelf UNet here, for example pytorch 2D UNet importieren, unet so umformen ggf., dass es in loadable wrapped st.
        # self.unet=UNet2D(in_channels=1)
        # self.unet=UNet2DModel(in_channels=1, out_channels=1, sample_size=(256,256))
        print("Initialized UNet")

    def forward(self, x, frozen=False):
        with torch.set_grad_enabled(
            not frozen
        ):  # If frozen True then use torch.no_grad
            # @Niklas: TODO add the forward pass here
            # forward x=UNet(x)
            # TODO is there a whole pass through both or just the SAM for cross entropy
            # x=x[0:1,:,0:256,0:256]
            # x=x.permute(0,2,3,1)
            print(x.shape)

            # print("out:")
            # x=x.to("cpu")
            print(x.device)
            out = self.unet.forward(sample=x, timestep=0)
            print("out")
            return out
            # return x
