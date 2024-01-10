from hashlib import sha512
import os
from typing import List
import time
import requests

import torch
from cog import BasePredictor, Input, Path
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    PNDMScheduler,
    LMSDiscreteScheduler,
    DDIMScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)
import numpy as np

from lora_diffusion import LoRAManager, monkeypatch_remove_lora
from t2i_adapters import Adapter
from t2i_adapters import patch_pipe as patch_pipe_t2i_adapter
from PIL import Image

import dotenv
import os

dotenv.load_dotenv()

MODEL_ID = os.environ.get("MODEL_ID", None)
MODEL_CACHE = "diffusers-cache"
SAFETY_MODEL_ID = os.environ.get("SAFETY_MODEL_ID", None)
IS_FP16 = os.environ.get("IS_FP16", "0") == "1"


def url_local_fn(url):
    filename = url.split("/")[-1]
    # Generate a unique 3-digit hash from the URL
    hash_digits = sha512(url.encode()).hexdigest()[:3]
    return f"{filename}_{hash_digits}.safetensors"
    
def download_lora(url):
    # TODO: allow-list of domains

    fn = url_local_fn(url)

    if not os.path.exists(fn):
        print("Downloading LoRA model... from", url)
        # stream chunks of the file to disk
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(fn, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

    else:
        print("Using disk cache...")

    return fn

def load_lora(lora_paths, pipe_tmp):
  for lora_name in lora_paths:
    lora_name_no_extension= os.path.splitext(lora_name)[0].replace(".","") # adapter name can't have '.' dot.
    print (lora_name, lora_name_no_extension)
    pipe_tmp.load_lora_weights(".", weight_name=lora_name, adapter_name=lora_name_no_extension)

def set_lora_weights(lora_paths, pipe_tmp, lora_scales):
  lora_weight_list = lora_scales
  lora_names = [os.path.splitext(lora_i_path)[0].replace(".","") for lora_i_path in lora_paths ]
  print(lora_names)
  print (lora_weight_list)
  exec(f"pipe_tmp.set_adapters({lora_names}, adapter_weights=lora_weight_list)")


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        self.pipe = StableDiffusionPipeline.from_pretrained(
            MODEL_CACHE,
            torch_dtype=torch.float16 if IS_FP16 else torch.float32,
        ).to("cuda")

        self.img2img_pipe = StableDiffusionImg2ImgPipeline(
            vae=self.pipe.vae,
            text_encoder=self.pipe.text_encoder,
            tokenizer=self.pipe.tokenizer,
            unet=self.pipe.unet,
            scheduler=self.pipe.scheduler,
            safety_checker=self.pipe.safety_checker,
            feature_extractor=self.pipe.feature_extractor,
        ).to("cuda")

        self.token_size_list: list = []
        self.ranklist: list = []
        self.loaded = None
        self.lora_manager = None

    def set_lora(self, urllists: List[str], scales: List[float]):
        # assert len(urllists) == len(scales), "Number of LoRAs and scales must match."

        # merged_fn = url_local_fn(f"{'-'.join(urllists)}")

        # if self.loaded == merged_fn:
        #     print("The requested LoRAs are loaded.")
        #     assert self.lora_manager is not None
        # else:

        #     st = time.time()
        #     self.lora_manager = LoRAManager(
        #         [download_lora(url) for url in urllists], self.pipe
        #     )
        #     self.loaded = merged_fn
        #     print(f"merging time: {time.time() - st}")

        # self.lora_manager.tune(scales)
        print("set_lora disabled!")

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(
            description="Input prompt. Use <1>, <2>, <3>, etc., to specify LoRA concepts",
            default="a photo of <1> riding a horse on mars",
        ),
        negative_prompt: str = Input(
            description="Specify things to not see in the output",
            default="",
        ),
        width: int = Input(
            description="Width of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        height: int = Input(
            description="Height of output image. Maximum size is 1024x768 or 768x1024 because of memory limits",
            choices=[128, 256, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024],
            default=512,
        ),
        num_outputs: int = Input(
            description="Number of images to output.",
            ge=1,
            le=4,
            default=1,
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=25
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        image: Path = Input(
            description="(Img2Img) Inital image to generate variations of. If this is not none, Img2Img will be invoked.",
            default=None,
        ),
        prompt_strength: float = Input(
            description="(Img2Img) Prompt strength when providing the image. 1.0 corresponds to full destruction of information in init image",
            default=0.8,
        ),
        scheduler: str = Input(
            default="DPMSolverMultistep",
            choices=[
                "DDIM",
                "K_EULER",
                "DPMSolverMultistep",
                "K_EULER_ANCESTRAL",
                "PNDM",
                "KLMS",
            ],
            description="Choose a scheduler.",
        ),
        lora_urls: str = Input(
            description="List of urls for safetensors of lora models, seperated with | .",
            default="",
        ),
        lora_scales: str = Input(
            description="List of scales for safetensors of lora models, seperated with | ",
            default="0.5",
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
        adapter_condition_image: Path = Input(
            description="(T2I-adapter) Adapter Condition Image to gain extra control over generation. If this is not none, T2I adapter will be invoked. Width, Height of this image must match the above parameter, or dimension of the Img2Img image.",
            default=None,
        ),
        adapter_type: str = Input(
            description="(T2I-adapter) Choose an adapter type for the additional condition.",
            choices=["sketch", "seg", "keypose", "depth"],
            default="sketch",
        ),
        clip_skip: int = Input(
            description="clip skip value. Leave blank to randomize the seed", default=1
        ),
    ) -> List[Path]:


        """Run a single prediction on the model"""
        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        if image is not None:
            pil_image = Image.open(image).convert("RGB")
            width, height = pil_image.size

        print(f"Generating image of {width} x {height} with prompt: {prompt}")

        if width * height > 786432:
            raise ValueError(
                "Maximum size is 1024x768 or 768x1024 pixels, because of memory limits. Please select a lower width or height."
            )

        generator = torch.Generator("cuda").manual_seed(seed)
        # CUSTOM CODE HERE.
        if len(lora_urls) > 0:
            lora_urls = [u.strip() for u in lora_urls.split("|")]
            lora_scales = [float(s.strip()) for s in lora_scales.split("|")]
            urllists = lora_urls
            print (f"urllists: {urllists}")
            lora_paths = [download_lora(url) for url in urllists]
            load_lora(lora_paths=lora_paths, pipe_tmp=self.pipe)
            set_lora_weights(lora_paths=lora_paths, pipe_tmp=self.pipe, lora_scales=lora_scales)


        else:
            print("No LoRA models provided, using default model...")

        
        # either text2img or img2img
        if image is None:
            self.pipe.scheduler = make_scheduler(scheduler, self.pipe.scheduler.config)

            output = self.pipe(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs
                if negative_prompt is not None
                else None,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                clip_skip=clip_skip,
            )
        else:
            extra_kwargs = {
                "image": pil_image,
                "strength": prompt_strength,
            }

            self.img2img_pipe.scheduler = make_scheduler(
                scheduler, self.pipe.scheduler.config
            )

            output = self.img2img_pipe(
                prompt=[prompt] * num_outputs if prompt is not None else None,
                negative_prompt=[negative_prompt] * num_outputs
                if negative_prompt is not None
                else None,
                guidance_scale=guidance_scale,
                generator=generator,
                num_inference_steps=num_inference_steps,
                cross_attention_kwargs={"scale": 1.0}, ## added this param, not necessary but just in case.
                **extra_kwargs,
            )
        ## reset lora weights and adapters so when invoked next time, no duplicate lora is loaded.
        self.pipe.unload_lora_weights() 
        output_paths = []
        for i, sample in enumerate(output.images):
            if output.nsfw_content_detected and output.nsfw_content_detected[i]:
                continue

            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        if len(output_paths) == 0:
            raise Exception(
                "NSFW content detected. Try running it again, or try a different prompt."
            )

        return output_paths


def make_scheduler(name, config):
    return {
        "PNDM": PNDMScheduler.from_config(config),
        "KLMS": LMSDiscreteScheduler.from_config(config),
        "DDIM": DDIMScheduler.from_config(config),
        "K_EULER": EulerDiscreteScheduler.from_config(config),
        "K_EULER_ANCESTRAL": EulerAncestralDiscreteScheduler.from_config(config),
        "DPMSolverMultistep": DPMSolverMultistepScheduler.from_config(config),
    }[name]
