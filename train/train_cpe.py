# ref:
# - https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L566
# - https://huggingface.co/spaces/baulab/Erasing-Concepts-In-Diffusion/blob/main/train.py
# - https://github.com/p1atdev/LECO/blob/main/train_lora.py
# - https://github.com/Con6924/SPM

import argparse
from pathlib import Path
import gc
from copy import deepcopy
import pandas as pd
import random

import numpy as np
import torch
from tqdm import tqdm
import os, sys
import numpy as np

sys.path[0] = "/".join(sys.path[0].split('/')[:-1])

from src.models.merge_cpe import *
from src.models.cpe_resag import(
    CPELayer_ResAG,
    CPENetwork_ResAG
)
from src.models.cpe_adv_memory import(
    PromptTuningLayer
)

from src.engine.sampling import sample, AnchorSamplerGensim
import src.engine.train_util as train_util
from src.models import model_util
from src.configs import config as config_pkg
from src.configs import prompt as prompt_pkg
from src.configs.config import RootConfig
from src.configs.prompt import PromptEmbedsCache, PromptEmbedsPair, PromptSettings
from src.engine.trainer import *

import wandb

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= "0"
DEVICE_CUDA = torch.device("cuda")


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


def train(
    config: RootConfig,
    prompts: list[PromptSettings],
):
    metadata = {
        "prompts": ",".join([prompt.json() for prompt in prompts]),
        "config": config.json(),
    }
    model_metadata = {
        "prompts": ",".join([prompt.target for prompt in prompts]),
        "rank": str(config.network.rank),
        "alpha": str(config.network.alpha),
    }
    save_path = Path(config.save.path)

    if config.logging.verbose:
        print(metadata)

    weight_dtype = config_pkg.parse_precision(config.train.precision)
    save_weight_dtype = config_pkg.parse_precision(config.train.precision)

        
    if config.logging.use_wandb:
        wandb.init(project=config.logging.project_name, 
                   config=metadata, 
                   name=config.logging.run_name+"_"+prompts[0].target.replace(' ', '_'), 
                   settings=wandb.Settings(symlink=False))

    (
        tokenizer, 
        text_encoder, 
        unet, 
        noise_scheduler, 
        pipe
    ) = model_util.load_models(
        config.pretrained_model.name_or_path,
        scheduler_name=config.train.noise_scheduler,
        v2=config.pretrained_model.v2,
        v_pred=config.pretrained_model.v_pred,
        device= DEVICE_CUDA
    )


    text_encoder.to(DEVICE_CUDA, dtype=weight_dtype)
    text_encoder.eval()

    unet.to(DEVICE_CUDA, dtype=weight_dtype)
    unet.enable_xformers_memory_efficient_attention()
    unet.requires_grad_(False)
    unet.eval()
    
    text_encoder.requires_grad_(False)
    text_encoder.eval()
    pipe.safety_checker = None
    
    
    anchor_sampler = AnchorSamplerGensim()
    
    task_id = len(config.pretrained_model.safetensor)
    
    network = CPENetwork_ResAG(
        unet,
        text_encoder,
        rank=config.network.rank,
        multiplier=1.0,
        alpha=config.network.alpha,
        module=CPELayer_ResAG,
        continual=True,
        task_id=task_id,
        continual_rank=config.network.continual_rank,
        hidden_size=config.network.hidden_size,
        init_size=config.network.init_size,
    ).to(DEVICE_CUDA, dtype=weight_dtype)    
        
    print()
    print("gate rank of netowrk:" , config.network.init_size)
    print("Prompts")
    for settings in prompts:
        print(settings)
    print("pal:", config.train.pal)
    print()

    prompt_scripts_path = config.scripts_file
    prompt_scripts_df = pd.read_csv(prompt_scripts_path)
    prompt_scripts_list = prompt_scripts_df['prompt'].to_list()
    replace_word = config.replace_word
    prompt_scripts_list += [replace_word]*(1)

    
    ########################################################
    ######## Compute lipschitz for weight matrices #########
    lipschitz_o = []
    lipschitz_q = []
    lipschitz_ov = []
    
    unet_modules = dict()
    for name, module in unet.named_modules():
        if ("attn2" in name) and (module.__class__.__name__ == "Attention"):
            mat_o = module.to_out[0].weight.detach()
            mat_q = module.to_q.weight.detach()
            # mat_k= module.to_k.weight.detach()
            mat_v = module.to_v.weight.detach()
            
            _, S_o, _ = torch.svd(mat_o)
            _, S_q, _ = torch.svd(mat_q)
            _, S_ov, _ = torch.svd(mat_o @ mat_v)
            
            lipschitz_o.append(S_o[0])
            lipschitz_q.append(S_q[0])
            lipschitz_ov.append(S_ov[0])
    lipschitz = {"lipschitz_o": lipschitz_o, "lipschitz_q": lipschitz_q, "lipschitz_ov": lipschitz_ov}
    ######## Compute lipschitz for weight matrices #########
    ########################################################
    
    ########################################################        
    ###################### Prepare #########################
    embedding_unconditional = train_util.encode_prompts(tokenizer, text_encoder, [""])
    
    network_modules = dict()
    for name, module in network.named_modules():
        if "CPELayer" in module.__class__.__name__:
            network_modules[name] = module

    unet_modules = dict()
    for name, module in unet.named_modules():
        name = "_".join(name.split("."))
        name = "lora_unet_" + name

        for network_name in network_modules.keys():
            if name == network_name:
                unet_modules[name] = module    
    ###################### Prepare #########################
    ########################################################        
    
    _, num_tokens, token_dim = embedding_unconditional.size()    
    adv_prompts = PromptTuningLayer(config.train.num_add_prompts, \
                                    num_tokens, token_dim, DEVICE_CUDA, weight_dtype\
                                   ).to(DEVICE_CUDA, dtype=weight_dtype)
    

    if config.train.resume_stage > 0:
        print("########################################")
        print(f"###### Resuming from stage {config.train.resume_stage} #######")
        print("########################################")
        
        model_path = save_path / f"{config.save.name}_stage{int(config.train.resume_stage)}.safetensors"
        model_path_adv = save_path / f"{config.save.name}_adv_prompts_stage{int(config.train.resume_stage)}.safetensors"

        cpes, metadatas = load_state_dict(model_path, weight_dtype) 
        for k, v in network.named_parameters(): 
            v.data=cpes[k].to(DEVICE_CUDA)
    
        cpes_adv, metadatas_adv = load_state_dict(model_path_adv, weight_dtype)
        for _ in range(config.train.resume_stage):
            adv_prompts.expand_prompts()
        adv_prompts.load_state_dict(cpes_adv)
        
    
    ####################################################################
    ############### Prepare for erasing token cache ####################
    prompt_one = prompts

    cache = dict()      
    with torch.no_grad():            
        prompt_pairs: list[PromptEmbedsPair] = []
        
        prompt_in_scripts_target = []
        prompt_in_scripts_neutral = []
        
        for settings in prompt_one:
            for prompt_script in prompt_scripts_list:
                pr_in_script_tgt = prompt_script.replace(replace_word, settings.target)
                pr_in_script_tgt = pr_in_script_tgt.replace(replace_word.lower(), settings.target)

                pr_in_script_ntl = prompt_script.replace(replace_word, settings.neutral)
                pr_in_script_ntl = pr_in_script_ntl.replace(replace_word.lower(), settings.neutral)                
                

                prompt_in_scripts_target.append(pr_in_script_tgt)
                prompt_in_scripts_neutral.append(pr_in_script_ntl)
            
        embeddings_erase_tgt = train_util.encode_prompts(
                    tokenizer, text_encoder, prompt_in_scripts_target).unsqueeze(1)

        embeddings_erase_ntl = train_util.encode_prompts(
                    tokenizer, text_encoder, prompt_in_scripts_neutral).unsqueeze(1)

        embeddings_erase_cache = torch.cat([embeddings_erase_tgt, embeddings_erase_ntl], dim=1)
            
    ############### Prepare for erasing token cache ####################
    ####################################################################   
    
    
    
    
    ######################################################################
    ############### Prepare for anchornig token cache ####################

    samples = []        
    simWords = []

    ########## for debugging with sim words from csv ############
    with torch.no_grad():
        if replace_word == "explicit":
            simWords_csv = pd.read_csv("./anchors/mass_surr_prompts_explicit.csv")
            simWords = list(simWords_csv.itertuples(index=False, name=None))
            simWords = [(a,b) for _,a,b in simWords]

            embeddings_anchor_cache = []
            prompt_in_scripts_anchor = [simWord[0] for simWord in simWords]
            embeddings_anchor_cache = train_util.encode_prompts(
                            tokenizer, text_encoder, prompt_in_scripts_anchor)

        else:
            if replace_word == "target":
                simWords_csv = pd.read_csv("./anchors/mass_surr_words_character.csv")
            elif replace_word == "actor" or replace_word == "artist":
                
                ###########################################################
                ##################### prepare simWords ####################
                if replace_word == "actor":
                    simWords_erase = [pr.target for pr in prompt_one]
                
                    simWords_general_words_csv = pd.read_csv("./anchors/mass_surr_words_actor_general_words.csv")
                    simWords_general_words = list(simWords_general_words_csv.itertuples(index=False, name=None))
                    simWords_general_words = [a for _,a,b in simWords_general_words]
        
                    simWords_actor_anchor_csv = pd.read_csv("./anchors/mass_surr_words_500celebs.csv")
                    simWords_actor_anchor = list(simWords_actor_anchor_csv.itertuples(index=False, name=None))
                
                    simWords_actor_anchor = [a[0] for a in simWords_actor_anchor]

                elif replace_word == "artist":
                    simWords_erase = [pr.target for pr in prompt_one]
                
                    simWords_general_words_csv = pd.read_csv("./anchors/mass_surr_words_artist_general_words.csv")
                    simWords_general_words = list(simWords_general_words_csv.itertuples(index=False, name=None))
                    simWords_general_words = [a for _,a,b in simWords_general_words]
        
                    simWords_actor_anchor_csv = pd.read_csv("./anchors/mass_surr_words_1734artists.csv")
                    simWords_actor_anchor = list(simWords_actor_anchor_csv.itertuples(index=False, name=None))
                    simWords_actor_anchor = [a[-1] for a in simWords_actor_anchor]
                    simWords_actor_anchor = list(set(simWords_actor_anchor))
                                                
                for simW_erase in simWords_erase:            
                    simWords_actor_anchor = [item for item in simWords_actor_anchor if simW_erase not in item.lower()]

                ##################### prepare simWords ####################
                ###########################################################

                #############################################################
                ##################### prepare embeddings ####################
                len_actor_anchor = len(simWords_actor_anchor)
                anchor_batch = 100
    
                simWords_act_anc_batch = []
                for batch_idx in range(int(math.ceil(float(len_actor_anchor)/anchor_batch))):
                    if anchor_batch*(batch_idx+1) <= len_actor_anchor:
                        simWords_act_anc_batch.append(simWords_actor_anchor[anchor_batch*batch_idx:anchor_batch*(batch_idx+1)])
                    else:
                        simWords_act_anc_batch.append(simWords_actor_anchor[anchor_batch*batch_idx:])
                embeddings_erase = train_util.encode_prompts(tokenizer, text_encoder, simWords_erase)            
    
                embeddings_actor_anchor = []
                for simW_batch in simWords_act_anc_batch:
                    emb_act_anc = train_util.encode_prompts(tokenizer, text_encoder, simW_batch)
                    embeddings_actor_anchor.append(emb_act_anc)
                embeddings_actor_anchor = torch.cat(embeddings_actor_anchor, dim=0)
                ##################### prepare embeddings ####################                
                #############################################################
                
                #############################################################
                ##################### compute similarity ####################
                emb_erase_flat = embeddings_erase.view(len(simWords_erase), -1)
                emb_anchor_flat = embeddings_actor_anchor.view(len(simWords_actor_anchor), -1)
    
                emb_erase_flat_norm = emb_erase_flat / emb_erase_flat.norm(2, dim=1, keepdim=True)
                emb_anchor_flat_norm = emb_anchor_flat / emb_anchor_flat.norm(2, dim=1, keepdim=True)
    
                similarity = emb_erase_flat_norm @ emb_anchor_flat_norm.T
                ##################### compute similarity ####################    
                #############################################################



                #############################################################
                #################### select anchor celebs ###################    
                val_sorted, ind_sorted = similarity.sort()
                ind_sorted_list = ind_sorted.cpu().numpy().tolist()
                
                simWords_selected = [simWords_actor_anchor[sim_idx] for sim_idx in ind_sorted_list[0][-50:]]
                #################### select anchor celebs ###################    
                #############################################################

                simWords = simWords_general_words + simWords_selected

            
            embeddings_anchor_cache = []
            for simWord in simWords:
                prompt_in_scripts_anchor = []

                for prompt_script in prompt_scripts_list:
                    pr_in_script_anc = prompt_script.replace(replace_word, simWord)
                    pr_in_script_anc = pr_in_script_anc.replace(replace_word.lower(), simWord)        
                    prompt_in_scripts_anchor.append(pr_in_script_anc)

                embeddings_erase_anc = train_util.encode_prompts(
                            tokenizer, text_encoder, prompt_in_scripts_anchor)

                embeddings_anchor_cache.append(embeddings_erase_anc)
            
            embeddings_anchor_cache = torch.cat(embeddings_anchor_cache, dim=0)

    ############### Prepare for anchoring token cache ####################
    ######################################################################   


    ######################################################################    
    #################### Generate original image  ########################
    if config.logging.gen_init_img:
        network.eval()
        
        log_dict = {}
        print("Generating samples...")
        with network:
            samples = train_util.text2img(
                pipe,
                prompts=config.logging.prompts,
                negative_prompt=config.logging.negative_prompt,
                width=config.logging.width,
                height=config.logging.height,
                num_inference_steps=config.logging.num_inference_steps,
                guidance_scale=config.logging.guidance_scale,
                generate_num=config.logging.generate_num,
                seed=config.logging.seed,
            )
        for text, img in samples:
            if len(text) > 30:
                text = text[:30]+text[-10:]
            log_dict[text+f"_stage{0}"] = wandb.Image(img)

        wandb.log(log_dict)
            
        network.train()    
    #################### Generate original image  ########################        
    ######################################################################    
    
    
    train_iterations = config.train.iterations
    train_iterations_adv = config.train.iterations_adv
    train_lr = config.train.lr
    train_lr_scheduler_num_cycles = config.train.lr_scheduler_num_cycles

    for stage in range(config.train.resume_stage, config.train.num_stages):      
        print(f"Stage: {stage}\n")
        
        if stage == 0:
            config.train.iterations = config.train.factor_init_iter*train_iterations
            config.train.lr = config.train.factor_init_lr*train_lr
            config.train.lr_scheduler_num_cycles = config.train.factor_init_lr_cycle*train_lr_scheduler_num_cycles
        else:
            config.train.iterations = train_iterations
            config.train.lr = train_lr
            config.train.lr_scheduler_num_cycles = train_lr_scheduler_num_cycles

        trainable_params = network.prepare_optimizer_params(
            config.train.text_encoder_lr, config.train.unet_lr, config.train.lr
        )

        pbar = tqdm(range(config.train.iterations))

        network.requires_grad_(True)
        network.train()
        adv_prompts.requires_grad_(False)
        adv_prompts.eval()
        
        optimizer_name, optimizer_args, optimizer = train_util.get_optimizer(
            config, trainable_params)
        lr_scheduler = train_util.get_scheduler_fix(config, optimizer)
        criteria = torch.nn.MSELoss()
                
        network, adv_prompts = train_erase_one_stage(
                stage, pbar, config, DEVICE_CUDA,
                pipe, unet, tokenizer, text_encoder,
                network, adv_prompts, network_modules, unet_modules,
                optimizer, lr_scheduler, criteria,
                prompt_scripts_list, prompts, replace_word, embedding_unconditional,
                anchor_sampler, lipschitz,
                save_weight_dtype,
                model_metadata,embeddings_erase_cache,embeddings_anchor_cache, ) 
        
        
        #########################################################      
        ####################### save ckpt #######################     
        if (stage+1)%config.save.stage_interval == 0:
            print("Saving Network...")

            save_path = Path(config.save.path)            
            save_path.mkdir(parents=True, exist_ok=True)
            network.save_weights(
                save_path / f"{config.save.name}_stage{stage+1}.safetensors",
                dtype=save_weight_dtype,
                metadata=model_metadata,
            )
            
            adv_prompts.save_weights(
                save_path / f"{config.save.name}_adv_prompts_last.safetensors",
                dtype=save_weight_dtype,
                metadata=model_metadata,
            )
        ####################### save ckpt #######################     
        #########################:30################################         

        
        if config.train.do_adv_learn:      
            adv_prompts.expand_prompts()
            
            pbar_adv = tqdm(range(config.train.iterations_adv))        
            network.requires_grad_(False)
            network.eval()
            adv_prompts.requires_grad_(True)
            adv_prompts.train()

            adv_parameters = adv_prompts.prompts.parameters()
            trainable_params_adv = [{"params": adv_parameters, "lr":config.train.lr_adv}]    

            optimizer_name_adv, optimizer_args_adv, optimizer_adv = train_util.get_optimizer(
                config, trainable_params_adv)
            lr_scheduler_adv = train_util.get_scheduler_adv(config, optimizer_adv)
            criteria_adv = torch.nn.MSELoss()        

            network, adv_prompts = train_adv_one_stage(
                    stage, pbar_adv, config, DEVICE_CUDA,
                    pipe, unet, tokenizer, text_encoder,
                    network, adv_prompts, network_modules, unet_modules,
                    optimizer_adv, lr_scheduler_adv, criteria_adv,
                    prompt_scripts_list, prompts, replace_word, embedding_unconditional, 
                    anchor_sampler, lipschitz,
                    save_weight_dtype,
                    model_metadata,
                    embeddings_erase_cache,
                    embeddings_anchor_cache)
      

    config.train.iterations = train_iterations
    config.train.lr = train_lr
    config.train.lr_scheduler_num_cycles = train_lr_scheduler_num_cycles        

    print("Saving...")
    save_path.mkdir(parents=True, exist_ok=True)
    network.save_weights(
        save_path / f"{config.save.name}_last.safetensors",
        dtype=save_weight_dtype,
        metadata=model_metadata,
    )
    
    adv_prompts.save_weights(
        save_path / f"{config.save.name}_adv_prompts_last.safetensors",
        dtype=save_weight_dtype,
        metadata=model_metadata,
    )

    
    del (
        unet,
        noise_scheduler,
        optimizer,
        network,
    )

    flush()
    wandb.finish()

    print("Done.")


def main(args):
    config_file = args.config_file

    config = config_pkg.load_config_from_yaml(config_file)
    prompts = prompt_pkg.load_prompts_from_yaml(config.prompts_file)
    
    if args.st_prompt_idx != -1:
        config.train.st_prompt_idx = args.st_prompt_idx
    if args.end_prompt_idx != -1:
        config.train.end_prompt_idx = args.end_prompt_idx
    if args.gate_rank != -1:
        config.network.init_size = args.gate_rank
        config.network.hidden_size = args.gate_rank
        config.network.continual_rank = args.gate_rank
    if args.guidance_scale != -1:
        for p_idx, p in enumerate(prompts):
            p.guidance_scale = args.guidance_scale 
    if args.pal != -1:
        config.train.pal = args.pal
    if args.resume_stage != -1:
        config.train.resume_stage = args.resume_stage
    if args.lora_rank != -1:
        config.network.rank = args.lora_rank     
    config.train.skip_learned = args.skip_learned
        
    exp_name = config.save.path.split("/")[-2].replace(f"guide#", f"guide{p.guidance_scale}").replace(f"pal#", f"pal{config.train.pal}").replace(f"gate_rank#", f"gate_rank{config.network.init_size}")

    if args.lora_rank != -1:
        exp_name = exp_name + "_lora_rank" + f"{config.network.rank}"    
    
    if len(args.noise) > 0:
        exp_name = exp_name + "_noise" + args.noise
        config.train.noise_scale = float(args.noise)

    if not args.mixup:
        exp_name = exp_name + "_nomixup"
        config.train.mixup = args.mixup
        
    config.save.path = "/".join(config.save.path.split("/")[:-2]+[exp_name]+[config.save.path.split("/")[-1]])

    base_path = config.save.path
    base_logging_prompts = config.logging.prompts
    
    for p_idx, p in enumerate(prompts):
        config.logging.prompts = [prompt.replace('[target]', p.target) if '[target]' in prompt else prompt for prompt in base_logging_prompts]
        config.save.path = base_path.replace(config.replace_word.upper(), p.target.replace(' ', '_'))
                
        if (p_idx < config.train.st_prompt_idx) or (p_idx > config.train.end_prompt_idx):
            continue
    
        os.makedirs(config.save.path, exist_ok=True)
        if config.train.skip_learned and os.path.isfile(f"{config.save.path}/{config.save.name}_last.safetensors"):
            print(f"{p_idx} {p.target} has already been trained")
            continue
                
        print()
        print(p_idx, [p])
        print("pal:", config.train.pal)
        print("noise_scale:", config.train.noise_scale)
        print()                
            
        seed_everything(config.train.train_seed)        
        train(config, [p])
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_file",
        required=True,
        help="Config file for training.",
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
    
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=-1,
    )
    
    parser.add_argument(
        "--pal",
        type=float,
        default=-1,
    )

    parser.add_argument(
        "--noise",
        type=str,
        default="",
    )


    
    parser.add_argument(
        "--resume_stage",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=-1,
    )
    
    parser.add_argument(
        "--mixup",
        type=bool,
        default=True,
    )
        
        
    parser.add_argument(
        "--skip_learned",
        type=bool,
        default=False,
    )
    

    args = parser.parse_args()
        
    main(args)
