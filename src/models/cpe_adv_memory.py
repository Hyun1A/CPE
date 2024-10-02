# ref:
# - https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py
# - https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py

import os
from copy import deepcopy
import math
from typing import Optional, List
import numpy as np
from src.models.merge_cpe import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPTextModel, CLIPTokenizer, CLIPTextModelWithProjection
from diffusers import UNet2DConditionModel
from safetensors.torch import save_file

import hashlib

from .cpe_resag import ParamModule


class PromptTuningLayer(nn.Module):
    def __init__(self, num_add_prompts, num_tokens, token_dim, device, weight_dtype):
        super(PromptTuningLayer, self).__init__()
        
        self.num_add_prompts = num_add_prompts
        self.num_tokens = num_tokens
        self.token_dim = token_dim        
        self.device = device
        self.weight_dtype = weight_dtype
        
        self.prompts = None # ParamModule(size)  # Zero-initialized learnable tensor
        self.prompts_prev = None

        self.len_prompts = 0
        self.len_prompts_prev = 0
        
    
    def expand_prompts(self, num_add_prompts=None):
        num_new_prompts = num_add_prompts if num_add_prompts is not None else self.num_add_prompts        
        self.len_prompts_prev += self.len_prompts
        self.len_prompts = num_new_prompts
        
        if self.prompts is not None and self.prompts_prev is not None:
            tmp = self.prompts_prev
            size_prev = tmp.weight.size() 
            self.prompts_prev = ParamModule(size=(size_prev[0] + self.prompts.weight.size(0), size_prev[1], size_prev[2])).to(self.device, self.weight_dtype)
            self.prompts_prev.weight.data[:tmp.weight.size(0)] = tmp.weight.data.detach()
            self.prompts_prev.weight.data[tmp.weight.size(0):] = self.prompts.weight.data.detach()
            

        elif self.prompts is not None and self.prompts_prev is None:
            self.prompts_prev = self.prompts

        # if self.prompts_prev is None:
        size = (num_new_prompts, self.num_tokens, self.token_dim)
        self.prompts = ParamModule(size=size).to(self.device, self.weight_dtype)
        nn.init.kaiming_uniform_(self.prompts.weight, a=math.sqrt(5))
        self.prompts.weight.data = self.prompts.weight.data / (self.token_dim**2)    

    def forward(self, x, idx=None):
        if idx is None:
            return x + self.prompts.weight
        else:
            return x + self.prompts.weight[idx]
        
    def forward_eval(self, x, idx=None):
        if idx is None:
            return x + self.prompts.weight.detach()
        else:
            return x + self.prompts.weight[idx].detach()
                
    
    def forward_prev(self, x, idx=None):
        if idx is None:
            return x + self.prompts_prev.weight
        else:
            return x + self.prompts_prev.weight[idx]    

    
    def forward_prev_eval(self, x, idx=None):
        if idx is None:
            return x + self.prompts_prev.weight.detach()
        else:
            return x + self.prompts_prev.weight[idx].detach()

    
    def save_weights(self, file, dtype=None, metadata: Optional[dict] = None):
        
        state_dict = self.state_dict()
        
        state_dict_save = dict()
        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict_save[key] = v
                
        if os.path.splitext(file)[1] == ".safetensors":
            save_file(state_dict_save, file, metadata)
        else:
            torch.save(state_dict_save, file)
