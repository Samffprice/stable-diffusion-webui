"""Generate API V1 routes

"""
from typing import List, Optional
from fastapi import APIRouter, Depends, Header, Query, UploadFile
from fastapi.params import File
from modules.util import HWC3

from fooocusapi.models.common.base import DescribeImageType
from fooocusapi.utils.api_utils import api_key_auth

from fooocusapi.models.common.requests import CommonRequest as Text2ImgRequest
from fooocusapi.models.requests_v1 import (
    ImgUpscaleOrVaryRequest,
    ImgPromptRequest,
    ImgInpaintOrOutpaintRequest
)
from fooocusapi.models.common.response import (
    AsyncJobResponse,
    GeneratedImageResult,
    DescribeImageResponse,
    StopResponse
)
from fooocusapi.utils.call_worker import call_worker
from fooocusapi.utils.img_utils import read_input_image
from fooocusapi.configs.default import img_generate_responses
from fooocusapi.worker import process_stop


secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)


def stop_worker():
    """Interrupt worker process"""
    process_stop()


@secure_router.post(
        path="/v1/generation/text-to-image",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV1"])
def text2img_generation(
    req: Text2ImgRequest,
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nText to Image Generation\n
    A text to image generation endpoint
    Arguments:
        req {Text2ImgRequest} -- Text to image generation request
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post(
        path="/v1/generation/image-upscale-vary",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV1"])
def img_upscale_or_vary(
    input_image: UploadFile,
    req: ImgUpscaleOrVaryRequest = Depends(ImgUpscaleOrVaryRequest.as_form),
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nImage upscale or vary\n
    Image upscale or vary
    Arguments:
        input_image {UploadFile} -- Input image file
        req {ImgUpscaleOrVaryRequest} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post(
        path="/v1/generation/image-inpaint-outpaint",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV1"])
def img_inpaint_or_outpaint(
    input_image: UploadFile,
    req: ImgInpaintOrOutpaintRequest = Depends(ImgInpaintOrOutpaintRequest.as_form),
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nInpaint or outpaint\n
    Inpaint or outpaint
    Arguments:
        input_image {UploadFile} -- Input image file
        req {ImgInpaintOrOutpaintRequest} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post(
        path="/v1/generation/image-prompt",
        response_model=List[GeneratedImageResult] | AsyncJobResponse,
        responses=img_generate_responses,
        tags=["GenerateV1"])
def img_prompt(
    cn_img1: Optional[UploadFile] = File(None),
    req: ImgPromptRequest = Depends(ImgPromptRequest.as_form),
    accept: str = Header(None),
    accept_query: str | None = Query(
        None, alias='accept',
        description="Parameter to override 'Accept' header, 'image/png' for output bytes")):
    """\nImage Prompt\n
    Image Prompt
    A prompt-based image generation.
    Arguments:
        cn_img1 {UploadFile} -- Input image file
        req {ImgPromptRequest} -- Request body
        accept {str} -- Accept header
        accept_query {str} -- Parameter to override 'Accept' header, 'image/png' for output bytes
    Returns:
        Response -- img_generate_responses
    """
    if accept_query is not None and len(accept_query) > 0:
        accept = accept_query

    return call_worker(req, accept)


@secure_router.post(
        path="/v1/tools/describe-image",
        response_model=DescribeImageResponse,
        tags=["GenerateV1"])
def describe_image(
    image: UploadFile,
    image_type: DescribeImageType = Query(
        DescribeImageType.photo,
        description="Image type, 'Photo' or 'Anime'")):
    """\nDescribe image\n
    Describe image, Get tags from an image
    Arguments:
        image {UploadFile} -- Image to get tags
        image_type {DescribeImageType} -- Image type, 'Photo' or 'Anime'
    Returns:
        DescribeImageResponse -- Describe image response, a string
    """
    if image_type == DescribeImageType.photo:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        interrogator = default_interrogator_photo
    else:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        interrogator = default_interrogator_anime
    img = HWC3(read_input_image(image))
    result = interrogator(img)
    return DescribeImageResponse(describe=result)

from extras.inpaint_mask import generate_mask_from_image

import io
import numpy as np
import base64
from PIL import Image
from io import BytesIO

@secure_router.post(
    path="/v1/tools/generate-mask",
    tags=["GenerateV1"])
async def generate_mask_route(
        image: UploadFile,
        mask_model: Optional[str] = 'isnet-general-use',
        cloth_category: Optional[str] = "full",
        sam_prompt_text: Optional[str] = None,
        sam_model: Optional[str] = 'sam_vit_b_01ec64',
        sam_quant: Optional[str] = 'False',
        box_threshold: Optional[float] = .3,
        text_threshold: Optional[float] = .25, ):
    """
    Generate a mask from an image
    ...
    """
    image_data = await image.read()
    image_data = Image.open(io.BytesIO(image_data))
    image_data = np.array(image_data)

    extras = {}
    if mask_model == 'u2net_cloth_seg':
        extras['cloth_category'] = cloth_category
    elif mask_model == 'sam':
        extras['sam_prompt_text'] = sam_prompt_text
        extras['sam_model'] = sam_model
        extras['sam_quant'] = sam_quant
        extras['box_threshold'] = box_threshold
        extras['text_threshold'] = text_threshold

    mask = generate_mask_from_image(image_data, mask_model, extras)

    # Convert the mask to a base64 string
    mask_image = Image.fromarray(mask)
    buffered = BytesIO()
    mask_image.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode()

    return {"filename": image.filename, "mask": mask_base64}
    
@secure_router.post(
        path="/v1/generation/stop",
        response_model=StopResponse,
        description="Job stopping",
        tags=["Default"])
def stop():
    """Interrupt worker"""
    stop_worker()
    return StopResponse(msg="success")
