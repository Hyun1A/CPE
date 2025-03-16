from pathlib import Path
import gc
from copy import deepcopy
import pandas as pd
import random

import torch
from tqdm import tqdm
import os, sys
import numpy as np

sys.path[0] = "/".join(sys.path[0].split('/')[:-1])


from src.engine.sampling import sample, AnchorSamplerGensim
import src.engine.train_util as train_util
from src.models import model_util
from src.configs import config as config_pkg
from src.configs import prompt as prompt_pkg
from src.configs.config import RootConfig
from src.configs.prompt import PromptEmbedsCache, PromptEmbedsPair, PromptSettings

import wandb

def train_erase_one_stage(
            stage,
            pbar,
            config,
            device_cuda,
            pipe,
            unet,
            tokenizer,
            text_encoder,
            network,
            adv_prompts,
            network_modules,
            unet_modules,
            optimizer,
            lr_scheduler,
            criteria,
            prompt_scripts_list,
            prompts,
            replace_word,
            embedding_unconditional,
            anchor_sampler,
            lipschitz,
            save_weight_dtype,
            model_metadata,
            embeddings_erase_cache,
            embeddings_anchor_cache,
            ):

    for i in pbar:
        loss=dict()
        optimizer.zero_grad()

        ####################################################################
        ################### Prepare for erasing prompt #####################
        prompt_one = prompts

        cache = dict()      
        with torch.no_grad():            
            prompt_pairs: list[PromptEmbedsPair] = []

            for settings in prompt_one:
                ind = random.randint(0, embeddings_erase_cache.size(0)-1)
                embeddings = embeddings_erase_cache[ind]

                cache[settings.target] = embeddings[0].unsqueeze(0)
                cache[settings.neutral] = embeddings[1].unsqueeze(0)
                cache['unconditional'] = embedding_unconditional

                prompt_pair = PromptEmbedsPair(
                    criteria,
                    cache[settings.target],
                    cache[settings.target],
                    cache['unconditional'],
                    cache[settings.neutral],
                    settings,
                )
                assert prompt_pair.sampling_batch_size % prompt_pair.batch_size == 0
                prompt_pairs.append(prompt_pair)

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

        ################### Prepare for erasing prompt #####################
        ####################################################################



        #####################################################################
        ################## Prepare for anchoring prompt #####################
        with torch.no_grad():
            anchors = anchor_sampler.sample_mixup_batch_cache(prompt_pair, tokenizer=tokenizer, \
                                    text_encoder=text_encoder, network=network, \
                                    prompt_scripts_list=prompt_scripts_list, replace_word=replace_word, \
                                    embeddings_anchor_cache=embeddings_anchor_cache, \
                                    scale=config.train.noise_scale, mixup=config.train.mixup)
            
        ################## Prepare for anchoring prompt #####################
        #####################################################################            


        
        #####################################################################
        ################### Prepare adversairal prompt ######################
        embedgings_adv = None
        len_emb_adv = 0
        if adv_prompts.len_prompts > 0:
            embeddings_adv = adv_prompts.forward_eval(prompt_pair.target)
            len_emb_adv = embeddings_adv.size(0)

        ################### Prepare adversairal prompt ######################
        #####################################################################        
        
        

        #########################################################
        ############### loss_prompt_erase/anchor ################   
        pal = torch.tensor([config.train.pal]).float().to(device=device_cuda)     

        pal_k_coef_log_dict_erase = dict()
        pal_v_coef_log_dict_erase = dict()        
        loss_prompt_erase_to_k = 0
        loss_prompt_erase_to_v = 0

        pal_k_coef_log_dict_anchor = dict()
        pal_v_coef_log_dict_anchor = dict()
        loss_prompt_anchor_to_k = 0
        loss_prompt_anchor_to_v = 0   

        loss_adv_erase_to_k = torch.tensor([0.]).float().to(device=device_cuda)     
        loss_adv_erase_to_v = torch.tensor([0.]).float().to(device=device_cuda)     
        
        idx = 0
    
        for name in network_modules.keys():
            if not "lora_unet" in name:
                continue
            if "lora_adaptor" in name:
                continue

            targets = torch.cat([prompt_pair.target, embeddings_adv]) if len_emb_adv > 0 else prompt_pair.target

            with torch.no_grad():
                crsattn_org = unet_modules[name](torch.cat([targets, prompt_pair.neutral, prompt_pair.unconditional, anchors[1::2]], dim=0).float())
                crsattn_target_org = crsattn_org[0].unsqueeze(0) #if targets.size(0) == 1 else crsattn_org[:1+len_emb_adv]
                crsattn_neutral_org = crsattn_org[1+len_emb_adv].unsqueeze(0)
                crsattn_comp_org = crsattn_org[(2+len_emb_adv):]
                
                if len_emb_adv > 0:
                    crsattn_target_adv_org = crsattn_org[1:1+len_emb_adv]
                    crsattn_neutral_adv_org = crsattn_org[1+len_emb_adv].unsqueeze(0).repeat(len_emb_adv,1,1)
                

            with network:
                crsattn = unet_modules[name](torch.cat([targets, prompt_pair.neutral, prompt_pair.unconditional, anchors[1::2]], dim=0).float())
                crsattn_target = crsattn[0].unsqueeze(0) #if targets.size(0) == 1 else crsattn[:1+len_emb_adv]
                crsattn_comp = crsattn[(2+len_emb_adv):]
                
                if len_emb_adv > 0:
                    crsattn_target_adv = crsattn[1:1+len_emb_adv]
                    crsattn_neutral = crsattn[1+len_emb_adv].unsqueeze(0).repeat(1+len_emb_adv,1,1)


            g_scale = prompt_pair.guidance_scale
            if "to_k" in name:
                lipschitz_for_key_target = ( lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx] ).unsqueeze(0).unsqueeze(1).unsqueeze(2) / prompt_pair.target.shape[1]
                loss_prompt_erase_to_k += (lipschitz_for_key_target * ( (crsattn_neutral_org - crsattn_target) + g_scale*(crsattn_neutral_org-crsattn_target_org) )**2).mean()
                if len_emb_adv > 0:
                    loss_adv_erase_to_k += (lipschitz_for_key_target * ( (crsattn_neutral_adv_org - crsattn_target_adv) + g_scale*(crsattn_neutral_org-crsattn_target_org) )**2).mean()
                pal_k_coef_log_dict_erase[f"pal_k_coef_log_dict_erase/{idx}th-layer"] = lipschitz_for_key_target.mean()

                lipschitz_for_key_comp = ( lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx] ).unsqueeze(0).unsqueeze(1).unsqueeze(2) / prompt_pair.target.shape[1]
                loss_prompt_anchor_to_k += (lipschitz_for_key_comp * (crsattn_comp_org-crsattn_comp)**2).mean()
                pal_k_coef_log_dict_anchor[f"pal_k_coef_log_dict_anchor/{idx}th-layer"] = lipschitz_for_key_comp.mean()                

            else:
                lipschitz_for_val_target = lipschitz['lipschitz_o'][idx].unsqueeze(0).unsqueeze(1)
                loss_prompt_erase_to_v += (lipschitz_for_val_target * ( (crsattn_neutral_org - crsattn_target) + g_scale*(crsattn_neutral_org-crsattn_target_org) )**2).mean()
                if len_emb_adv > 0:
                    loss_adv_erase_to_v += (lipschitz_for_val_target * ( (crsattn_neutral_adv_org - crsattn_target_adv) + g_scale*(crsattn_neutral_org-crsattn_target_org) )**2).mean()
                pal_v_coef_log_dict_erase[f"pal_v_coef_log_dict_erase/{idx}th-layer"] = lipschitz_for_val_target.mean()

                lipschitz_for_val_comp = lipschitz['lipschitz_o'][idx].unsqueeze(0).repeat(crsattn_comp.shape[0],1).unsqueeze(2)
                loss_prompt_anchor_to_v += (lipschitz_for_val_comp * (crsattn_comp_org-crsattn_comp)**2).mean() #/ crsattn_comp_org.shape[0]
                pal_v_coef_log_dict_anchor[f"pal_v_coef_log_dict_anchor/{idx}th-layer"] = lipschitz_for_val_comp.mean()                

                idx+=1
        
        
        
        loss_prompt_erase_to_k = loss_prompt_erase_to_k / len(network_modules)
        loss_prompt_erase_to_v = loss_prompt_erase_to_v / len(network_modules)
        loss_prompt_erase = loss_prompt_erase_to_v + loss_prompt_erase_to_k       

        loss_prompt_anchor_to_k = loss_prompt_anchor_to_k / len(network_modules)
        loss_prompt_anchor_to_v = loss_prompt_anchor_to_v / len(network_modules)
        loss_prompt_anchor = loss_prompt_anchor_to_v + loss_prompt_anchor_to_k        


        loss_adv_erase_to_k = loss_adv_erase_to_k / len(network_modules)
        loss_adv_erase_to_v = loss_adv_erase_to_v / len(network_modules)
        loss_adv_erase = loss_adv_erase_to_v + loss_adv_erase_to_k


        
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase"] = loss_prompt_erase
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase_to_k"] = loss_prompt_erase_to_k
        loss[f"loss_erasing_stage{stage}/loss_prompt_erase_to_v"] = loss_prompt_erase_to_v

        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor"] = loss_prompt_anchor
        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor_to_k"] = loss_prompt_anchor_to_k
        loss[f"loss_erasing_stage{stage}/loss_prompt_anchor_to_v"] = loss_prompt_anchor_to_v 

        loss[f"loss_erasing_stage{stage}/loss_adv_erase"] = loss_adv_erase 
        
        adv_coef = config.train.adv_coef
        loss[f"loss_erasing"] = loss[f"loss_erasing_stage{stage}/loss_prompt_erase"] \
                            + adv_coef * loss[f"loss_erasing_stage{stage}/loss_adv_erase"] \
                            + pal * loss[f"loss_erasing_stage{stage}/loss_prompt_anchor"]

        ############### loss_prompt_erase/anchor ################        
        #########################################################                


        #########################################################        
        ######################### misc ##########################    
        loss["pal"] = pal
        loss["guidance"] = torch.tensor([prompt_pair.guidance_scale]).cuda()
        loss["la_strength"] = torch.tensor([prompt_pair.la_strength]).cuda()
        loss["batch_size"] = torch.tensor([prompt_pair.batch_size]).cuda()
        ######################### misc ##########################        
        #########################################################        

        #########################################################        
        ######################## optim ##########################        
        loss[f"loss_erasing"].backward()

        if config.train.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params, config.train.max_grad_norm, norm_type=2
            )
        optimizer.step()
        lr_scheduler.step()
        ######################## optim ##########################        
        #########################################################

        #########################################################
        ####################### logging #########################     
        pbar.set_description(f"Loss: {loss[f'loss_erasing'].item():.4f}")
        
        if config.logging.use_wandb:
            log_dict = {"iteration": i}
            loss = {k: v.detach().cpu().item() for k, v in loss.items()}
            log_dict.update(loss)
            lrs = lr_scheduler.get_last_lr()
            if len(lrs) == 1:
                log_dict["lr"] = float(lrs[0])
            else:
                log_dict["lr/textencoder"] = float(lrs[0])
                log_dict["lr/unet"] = float(lrs[-1])

            if config.train.optimizer_type.lower().startswith("dadapt"):
                log_dict["lr/d*lr"] = (
                    optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                )

            if (stage+1)%config.logging.stage_interval == 0 and config.logging.interval > 0 and i != 0 and (
                i % config.logging.interval == 0 or i == config.train.iterations - 1
            ):
                
                network.eval()
                
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
                    log_dict[text+f"_stage{stage+1}"] = wandb.Image(img)

                network.train()

                
            ################# additional log #################
            log_dict["info/rank"] = config.network.rank
            log_dict["info/num_embeddings"] = config.network.num_embeddings
            log_dict["info/batch_size"] = config.train.batch_size
            log_dict["info/hidden_size"] = config.network.hidden_size
            log_dict["info/init_size"] = config.network.init_size
            log_dict["info/adv_coef"] = adv_coef
            ################# additional log #################

            log_dict.update({k: v.detach().cpu().item() for k,v in pal_k_coef_log_dict_anchor.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_v_coef_log_dict_anchor.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_k_coef_log_dict_erase.items()})
            log_dict.update({k: v.detach().cpu().item() for k,v in pal_v_coef_log_dict_erase.items()})
            wandb.log(log_dict)

        ####################### logging #########################     
        #########################################################
    
    return network, adv_prompts, 





def train_adv_one_stage(
            stage,
            pbar_adv,
            config,
            device_cuda,
            pipe,
            unet,
            tokenizer,
            text_encoder,
            network,
            adv_prompts,
            network_modules,
            unet_modules,
            optimizer_adv,
            lr_scheduler_adv,
            criteria_adv,
            prompt_scripts_list,
            prompts,
            replace_word,
            embedding_unconditional,
            anchor_sampler,
            lipschitz,
            save_weight_dtype,
            model_metadata,
            embeddings_erase_cache,
            embeddings_anchor_cache,):

    for i in pbar_adv:
        loss_adv=dict()
        optimizer_adv.zero_grad()

        script_rand_idx = random.randint(0, len(prompt_scripts_list)-1)
        prompt_script = prompt_scripts_list[script_rand_idx]

        ####################################################################
        ################### Prepare for erasing prompt #####################
        prompt_one = prompts

        cache = dict()      
        with torch.no_grad():            
            prompt_pairs: list[PromptEmbedsPair] = []

            for settings in prompt_one:
                ind = random.randint(0, embeddings_erase_cache.size(0)-1)
                embeddings = embeddings_erase_cache[ind]

                cache[settings.target] = embeddings[0].unsqueeze(0)
                cache[settings.neutral] = embeddings[1].unsqueeze(0)
                cache['unconditional'] = embedding_unconditional

                prompt_pair = PromptEmbedsPair(
                    criteria_adv,
                    cache[settings.target],
                    cache[settings.target],
                    cache['unconditional'],
                    cache[settings.neutral],
                    settings,
                )
                assert prompt_pair.sampling_batch_size % prompt_pair.batch_size == 0
                prompt_pairs.append(prompt_pair)

            height, width = (
                prompt_pair.resolution,
                prompt_pair.resolution,
            )
            if prompt_pair.dynamic_resolution:
                height, width = train_util.get_random_resolution_in_bucket(
                    prompt_pair.resolution
                )

        ################### Prepare for erasing prompt #####################
        ####################################################################
        


        #####################################################################
        ################### Prepare adversairal prompt ######################
        embeddings_adv = adv_prompts.forward(prompt_pair.target)
        len_emb_adv = embeddings_adv.size(0)
        ################### Prepare adversairal prompt ######################
        #####################################################################

        #########################################################
        ################### loss_prompt_adv #####################            
        loss_prompt_adv_to_k = 0
        loss_prompt_adv_to_v = 0

        idx = 0
        for name in network_modules.keys():
            if not "lora_unet" in name:
                continue
            if "lora_adaptor" in name:
                continue

            targets = embeddings_adv

            with torch.no_grad():
                crsattn_target_org = unet_modules[name](prompt_pair.target).float().repeat(len_emb_adv,1,1)

            with network:
                crsattn_target_adv = unet_modules[name](targets).float()

            if "to_k" in name:
                lipschitz_for_key_target = ( lipschitz['lipschitz_ov'][idx]*lipschitz['lipschitz_q'][idx] ).unsqueeze(0).unsqueeze(1).unsqueeze(2) / prompt_pair.target.shape[1]
                loss_prompt_adv_to_k += (lipschitz_for_key_target * ( crsattn_target_adv - crsattn_target_org )**2).mean()

            else:
                lipschitz_for_val_target = lipschitz['lipschitz_o'][idx].unsqueeze(0).unsqueeze(1)
                loss_prompt_adv_to_v += (lipschitz_for_val_target * ( crsattn_target_adv - crsattn_target_org )**2).mean()

                idx+=1
                
        loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv_to_k"] = loss_prompt_adv_to_k / len(network_modules)
        loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv_to_v"] = loss_prompt_adv_to_v / len(network_modules)   
        loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv"] = loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv_to_k"] \
                                                            + loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv_to_v"]
        ################### loss_prompt_adv #####################        
        #########################################################            

        #########################################################        
        ######################## optim ##########################        
        loss_adv[f"loss_adv_stage{stage}/loss_prompt_adv"].backward()

        if config.train.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                trainable_params_adv, config.train.max_grad_norm, norm_type=2
            )
        optimizer_adv.step()
        lr_scheduler_adv.step()
        ######################## optim ##########################        
        #########################################################
        
        #########################################################
        ####################### logging #########################     
        pbar_adv.set_description(f"Loss: {loss_adv[f'loss_adv_stage{stage}/loss_prompt_adv'].item():.4f}")

        if config.logging.use_wandb:
            log_dict = {"iteration_adv": i}
            loss_adv = {k: v.detach().cpu().item() for k, v in loss_adv.items()}
            log_dict.update(loss_adv)
            lrs = lr_scheduler_adv.get_last_lr()
            if len(lrs) == 1:
                log_dict["lr_adv"] = float(lrs[0])
            else:
                log_dict["lr_adv/textencoder"] = float(lrs[0])
                log_dict["lr_adv/unet"] = float(lrs[-1])

            if config.train.optimizer_type.lower().startswith("dadapt"):
                log_dict["lr/d*lr"] = (
                    optimizer.param_groups[0]["d"] * optimizer.param_groups[0]["lr"]
                )

            wandb.log(log_dict)
        
    return network, adv_prompts