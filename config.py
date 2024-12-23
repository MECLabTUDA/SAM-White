mappings = {
    "700": "Task700_BraTS2020",
    "808": "Task808_mHeartA",
    "900": "Task900_BCSS",
    "300": "Task300_KvasirSEG",
}
train_on = [
    # ['700'],
    ["300"]
]

model_types = ["vit_b"]
checkpoints = ["checkpoints/sam_vit_b_01ec64.pth"]
device = 0

RESULT_PATH = "out"