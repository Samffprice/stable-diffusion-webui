# -*- coding: utf-8 -*-

""" File utils

Use for managing generated files

@file: file_utils.py
@author: Konie
@update: 2024-03-22
"""
import requests
import base64
import datetime
from io import BytesIO
import os
import json
from pathlib import Path
from google.cloud import storage
import datetime
import io
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from fooocusapi.utils.logger import logger


output_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), '../..', 'outputs', 'files'))
os.makedirs(output_dir, exist_ok=True)


def get_public_ip():
    response = requests.get('https://httpbin.org/ip')
    return response.json()['origin']


STATIC_SERVER_BASE = 'http://' + get_public_ip() + '/files/'


# STATIC_SERVER_BASE = 'http://127.0.0.1:7860/files/'


def save_output_file(
    img: np.ndarray,
    image_meta: dict = None,
    image_name: str = '',
    extension: str = 'png') -> str:
    """
    Save np image to Google Cloud Storage
    Args:
        img: np.ndarray image to save
        image_meta: dict of image metadata
        image_name: str of image name
        extension: str of image extension
    Returns:
        str of public url
    """
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")

    filename = os.path.join(date_string, image_name + '.' + extension)
    file_path = os.path.join(output_dir, filename)

    if extension not in ['png', 'jpg', 'webp']:
        extension = 'png'

    if image_meta is None:
        image_meta = {}

    meta = None
    if extension == 'png':
        meta = PngInfo()
        meta.add_text("parameters", json.dumps(image_meta))
        meta.add_text("fooocus_scheme", image_meta['metadata_scheme'])

    storage_client = storage.Client()
    bucket = storage_client.bucket('tla-filestorage')
    blob = bucket.blob(image_name)
    # Normalize the array in-place and convert to 8-bit unsigned integer
    # Normalize the array in-place and convert to 8-bit unsigned integer
    # Normalize the array and convert to 8-bit unsigned integer
    np_array = ((img - img.min()) * (1/(img.max() - img.min()) * 255)).astype('uint8')

    # Convert numpy array to image
    image = Image.fromarray(img)

    # Create a byte stream and save the image to it
    byte_stream = io.BytesIO()
    image.save(byte_stream, format=extension, pnginfo=meta, optimize=True)

    # Go to the start of the stream
    byte_stream.seek(0)

    # Set the chunk size to 5MB
    blob.chunk_size = 5 * 1024 * 1024  # 5MB

    # Upload the byte stream to Google Cloud Storage
    blob.upload_from_file(byte_stream, content_type='image/png')


    print(f"File {image_name} uploaded.")

    # Make the blob publicly accessible
    blob.make_public()

    # Get the public url
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    Image.fromarray(img).save(file_path, format=extension,
                              pnginfo=meta, optimize=True)
    return Path(filename).as_posix()


def delete_output_file(filename: str):
    """
    Delete files specified in the output directory
    Args:
        filename: str of file name
    """
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logger.std_warn(f'[Fooocus API] {filename} not exists or is not a file')
    try:
        os.remove(file_path)
        logger.std_info(f'[Fooocus API] Delete output file: {filename}')
    except OSError:
        logger.std_error(f'[Fooocus API] Delete output file failed: {filename}')


def output_file_to_base64img(filename: str | None) -> str | None:
    """
    Convert an image file to a base64 string.
    Args:
        filename: str of file name
    return: str of base64 string
    """
    if filename is None:
        return None
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None

    img = Image.open(file_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


def output_file_to_bytesimg(filename: str | None) -> bytes | None:
    """
    Convert an image file to a bytes string.
    Args:
        filename: str of file name
    return: bytes of image data
    """
    if filename is None:
        return None
    file_path = os.path.join(output_dir, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None

    img = Image.open(file_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    return byte_data


def get_file_serve_url(filename: str | None) -> str | None:
    """
    Get the static serve url of an image file.
    Args:
        filename: str of file name
    return: str of static serve url
    """
    if filename is None:
        return None
    return STATIC_SERVER_BASE + filename.replace('\\', '/')
