import argparse
import gc
from pathlib import Path
import pandas as pd
import torch
from typing import Literal
import sys
import random, os
import numpy as np


sys.path[0] = "/".join(sys.path[0].split('/')[:-1])

from src.configs.generation_config import load_config_from_yaml, GenerationConfig
from src.configs.config import parse_precision
from src.engine import train_util
from src.models import model_util
from src.models.cpe_resag import(
    CPELayer_ResAG,
    CPENetwork_ResAG
)

from src.models.merge_cpe import *
device = torch.device('cuda:0')
torch.cuda.set_device(device)

UNET_NAME = "unet"
TEXT_ENCODER_NAME = "text_encoder"



def seed_everything(seed: int):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True



def flush():
    torch.cuda.empty_cache()
    gc.collect()

def infer_with_cpe(
        model_path: list[str],
        config: GenerationConfig,
        base_model: str = "CompVis/stable-diffusion-v1-4",
        v2: bool = False,
        precision: str = "fp32",
    ):

    model_paths = model_path
    weight_dtype = parse_precision(precision)
        
    # load the pretrained SD
    tokenizer, text_encoder, unet, pipe = model_util.load_checkpoint_model(
        base_model,
        v2=v2,
        weight_dtype=weight_dtype
    )
    pipe.safety_checker = None
    pipe.requires_safety_checker = False
    special_token_ids = set(tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map.values()))

    text_encoder.to(device, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(device, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()

    cpes, metadatas = zip(*[
        load_state_dict(model_path, weight_dtype) for model_path in model_paths
    ])
        
    # check if CPEs are compatible
    assert all([metadata["rank"] == metadatas[0]["rank"] for metadata in metadatas])

    # get the erased concept
    erased_prompts = [md["prompts"].split(",") for md in metadatas]
    erased_prompts_count = [len(ep) for ep in erased_prompts]
    print(f"Erased prompts: {erased_prompts}")

    print(metadatas[0])
    network = CPENetwork_ResAG(
        unet,
        text_encoder,
        rank=int(float(metadatas[0]["rank"])),
        multiplier=1.0,
        alpha=float(metadatas[0]["alpha"]),
        module=CPELayer_ResAG,
        continual=True,
        task_id=10,
        continual_rank=config.gate_rank, 
        hidden_size=config.gate_rank,  
        init_size=config.gate_rank,  
        n_concepts=len(model_paths),
    ).to(device, dtype=weight_dtype)  
    
    for k,v in network.named_parameters(): 
        print(f"{k:100}", v.shape, cpes[0][k].shape)
        for idx in range(len(cpes)):
            if len(v.shape) > 1:
                v.data[idx,:] = cpes[idx][k]
            else:
                v.data[idx] = cpes[idx][k]
                
                
                
    network.to(device, dtype=weight_dtype)  

    
    network.eval()
    network.set_inference_mode()
    
    print(config.save_path)
    
    promptDf = pd.read_csv(config.prompt_path)
    with torch.no_grad():
        for p_idx, (k, row) in enumerate(promptDf.iterrows()):
            
            if (p_idx < config.st_prompt_idx) or (p_idx > config.end_prompt_idx):
                continue
            
            
            prompt = row['prompt']
            prompt += config.unconditional_prompt
            print(f"{p_idx}, Generating for prompt: {prompt}")
            prompt_embeds, prompt_tokens = train_util.encode_prompts(
                tokenizer, text_encoder, [prompt], return_tokens=True
                )
            
            print(os.path.isfile(config.save_path.format(prompt.replace(" ", "_"), row['evaluation_seed'], "0")))
            if os.path.isfile(config.save_path.format(prompt.replace(" ", "_"), row['evaluation_seed'], "0")):
                print(row)
                print('cont')  
                continue
                
            network.reset_cache_attention_gate()

            seed_everything(row['evaluation_seed'])
            
            with network:
                images = pipe(
                    negative_prompt=config.negative_prompt,
                    width=config.width,
                    height=config.height,
                    num_inference_steps=config.num_inference_steps,
                    guidance_scale=config.guidance_scale,
                    generator=torch.cuda.manual_seed(row['evaluation_seed']),
                    num_images_per_prompt=config.generate_num,
                    prompt_embeds=prompt_embeds,
                ).images
            folder = Path(config.save_path.format(prompt.replace(" ", "_"), "0", "0")).parent
            if not folder.exists():
                folder.mkdir(parents=True, exist_ok=True)
            for i, image in enumerate(images):
                image.save(
                    config.save_path.format(
                        prompt.replace(" ", "_"), row['evaluation_seed'], i
                    )
                )
            
            network.reset_cache_attention_gate()

def main(args):
    concepts_folder = os.listdir(args.model_path[0])    
    concepts_ckpt = []
    
    for folder in concepts_folder:
        for ckpt in os.listdir(os.path.join(args.model_path[0],folder)):
            if ("last.safetensors" in ckpt) and ("adv_prompts" not in ckpt):
                concepts_ckpt.append(os.path.join(args.model_path[0],folder,ckpt))

    model_path = [Path(lp) for lp in concepts_ckpt]
    
    generation_config = load_config_from_yaml(args.config)

    if args.st_prompt_idx != -1:
        generation_config.st_prompt_idx = args.st_prompt_idx
    if args.end_prompt_idx != -1:
        generation_config.end_prompt_idx = args.end_prompt_idx
    if args.gate_rank != -1:
        generation_config.gate_rank = args.gate_rank
    
    
    generation_config.save_path = os.path.join("/".join(generation_config.save_path.split("/")[:-3]), args.save_env, "/".join(generation_config.save_path.split("/")[-2:]))

    infer_with_cpe(
        model_path,
        generation_config,
        base_model=args.base_model,
        v2=args.v2,
        precision=args.precision,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/generation.yaml",
        help="Base configs for image generation.",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        nargs="*",
        help="CPE model to use.",
    )
    # model configs
    parser.add_argument(
        "--base_model",
        type=str,
        default="CompVis/stable-diffusion-v1-4",
        help="Base model for generation.",
    )
    parser.add_argument(
        "--v2",
        action="store_true",
        help="Use the 2.x version of the SD.",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        help="Precision for the base model.",
    )
    parser.add_argument(
        "--save_env",
        type=str,
        default="",
        help="Precision for the base model.",
    )    
    
    parser.add_argument(
        "--st_prompt_idx",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--end_prompt_idx",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--gate_rank",
        type=int,
        default=-1,
    )

    args = parser.parse_args()

    main(args)
