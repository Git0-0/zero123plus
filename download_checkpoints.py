import os
import warnings

import torch
import urllib.request
import huggingface_hub
from diffusers import DiffusionPipeline

def init_loader():
    warnings.simplefilter(action='ignore', category=FutureWarning)
    os.environ['HTTP_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

if 'HF_TOKEN' in os.environ:
    huggingface_hub.login(os.environ['HF_TOKEN'])
sam_checkpoint = "tmp/sam_vit_h_4b8939.pth"
os.makedirs('tmp', exist_ok=True)
urllib.request.urlretrieve(
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    sam_checkpoint
)
DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)
