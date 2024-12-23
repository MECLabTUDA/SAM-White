#!/usr/bin/env python3
import os
import hashlib
import shutil

import click
import requests
import tqdm
import numpy as np

import SimpleITK as sitk

from sklearn.model_selection import train_test_split

from PIL import Image

KVASIR_SEG_URL = "https://datasets.simula.no/downloads/kvasir-seg.zip"
KVASIR_SEG_CHECKSUM = "03b30e21d584e04facf49397a2576738fd626815771afbbf788f74a7153478f7"
RANDOM_STATE = 3299


def validate_checksum(filename: str):
    """
    Compares the SHA256 checksum of a file to a list of known checksums.

    :param filename: Filename as given in the above declared dictionaries.
    :ptype filename: str
    """
    checksum = KVASIR_SEG_CHECKSUM
    if checksum is None:
        click.secho(f"No checksum available to compare with.", fg="yellow")
        click.secho(f"The downloaded file might be unsafe.", fg="yellow")
        if not click.confirm("Still want to continue?"):
            return False
        return True
    else:
        click.secho(f"Validating checksum...", fg="blue")
        sha256 = hashlib.sha256()
        with open(filename, "rb") as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        file_hash = sha256.hexdigest()
        if checksum != file_hash:
            click.secho(
                f"Invalid checksum. This might be a bug, a server error or transmission problem.",
                fg="red",
            )
            click.secho(f"expected: {checksum}")
            click.secho(f"download: {file_hash}")
            return False
        click.secho("Checksum ok.", fg="green")
    return True


def download_kvasir_seg(filename: str):
    url = KVASIR_SEG_URL
    destination = filename
    click.secho(f"File: {filename}", fg="blue")
    click.secho(f"Downloading file from URL {url}...", fg="blue")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(destination, "wb") as f:
            for chunk in tqdm.tqdm(
                r.iter_content(chunk_size=1024),
                total=int(r.headers.get("content-length", 0)) // 1024,
                unit="KB",
            ):
                f.write(chunk)
    click.secho("Done.", fg="green")
    if not validate_checksum(filename):
        exit()
    click.secho("")


def extract_archive(filename, destination=None):
    """ """
    click.secho(f"Extracting archive {filename}...", fg="blue")
    archive_path = filename
    if not destination:
        destination = filename[: -len(".zip")]
    shutil.unpack_archive(archive_path, destination)
    click.secho("Done.", fg="green")


def download_and_extract():
    if os.path.exists("data/kvasir_seg"):
        return
    if os.path.exists("kvasir_seg.zip"):
        if not validate_checksum("kvasir_seg.zip"):
            download_kvasir_seg("kvasir_seg.zip")
            validate_checksum("kvasir_seg.zip")
    else:
        download_kvasir_seg("kvasir_seg.zip")
        validate_checksum("kvasir_seg.zip")
    extract_archive("kvasir_seg.zip", "data/kvasir_seg")


def scale256(image):
    return image.resize((256, 256))


def preprocess_image(image_filename, label_filename, image_dest, label_dest):
    image = Image.open(image_filename)
    label = Image.open(label_filename)

    bbox = image.getbbox()
    image = image.crop(bbox)
    label = label.crop(bbox)

    image = scale256(image)
    label = scale256(label)

    # shape: (256, 256, 3)   (256, 256, 3)

    image_array = np.asarray(image)
    label_array = np.asarray(label)[..., 0] > 0
    label_array = label_array.astype(np.uint8)

    # shape: (256, 256, 3)   (256, 256)

    image_array = np.expand_dims(image_array, axis=2)
    label_array = np.expand_dims(label_array, axis=2)

    # shape: (256, 256, 1, 3)   (256, 256, 1, 3)

    sitk_image = sitk.GetImageFromArray(image_array)
    sitk.WriteImage(
        sitk_image,
        os.path.join(image_dest[:-len(".jpg")] + ".png" + ".nii.gz"),
    )
    
    sitk_label = sitk.GetImageFromArray(label_array)
    sitk.WriteImage(
        sitk_label,
        os.path.join(label_dest[:-len(".jpg")] + ".png" + ".nii.gz"),
    )


def split_and_preprocess():
    os.makedirs("data/Task300_KvasirSEG", exist_ok=True)
    os.makedirs("data/Task300_KvasirSEG/imagesTr", exist_ok=True)
    os.makedirs("data/Task300_KvasirSEG/imagesTs", exist_ok=True)
    os.makedirs("data/Task300_KvasirSEG/labelsTr", exist_ok=True)
    os.makedirs("data/Task300_KvasirSEG/labelsTs", exist_ok=True)

    images_dir = "data/kvasir_seg/Kvasir-SEG/images"
    labels_dir = "data/kvasir_seg/Kvasir-SEG/masks"
    idx_images = np.arange(len(os.listdir(images_dir)))
    idx_train, idx_test = train_test_split(
        idx_images, test_size=0.2, random_state=RANDOM_STATE
    )

    click.secho("Preprocessing and sorting images...")
    for i, image_filename in tqdm.tqdm(
        enumerate(sorted(os.listdir(images_dir))), total=len(os.listdir(images_dir))
    ):
        image_path = os.path.join(images_dir, image_filename)
        label_path = os.path.join(labels_dir, image_filename)
        if i in idx_train:
            image_dest = "data/Task300_KvasirSEG/imagesTr/" + image_filename
            label_dest = "data/Task300_KvasirSEG/labelsTr/" + image_filename
        else:
            image_dest = "data/Task300_KvasirSEG/imagesTs/" + image_filename
            label_dest = "data/Task300_KvasirSEG/labelsTs/" + image_filename
        preprocess_image(image_path, label_path, image_dest, label_dest)


def run_all():
    download_and_extract()
    split_and_preprocess()


if __name__ == "__main__":
    run_all()
