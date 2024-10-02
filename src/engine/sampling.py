import random
import torch
import numpy as np
import pickle
import pandas as pd
import os
import gensim.downloader, random
import src.engine.train_util as train_util
from src.configs.prompt import PromptEmbedsPair
import pandas as pd

class AnchorSamplerGensim():
    def __init__(self, gensim_base:str = 'glove-twitter-100'):
        # path는 임의로 줌
        if not os.path.isfile(f'./{gensim_base}'):
            print(r'======== Downloading {} word2Vec model ========')
            self.wordVec = gensim.downloader.load('glove-twitter-100')
            pickle.dump(self.wordVec, open(f'./{gensim_base}','wb'))
            print(r'======== Complete ! {} word2Vec model ========')
        else:
            self.wordVec = pickle.load(open(f'./{gensim_base}', 'rb'))
    
    def sample_mixup_batch_cache(self, prompt_pair: PromptEmbedsPair, tokenizer=None,\
                               text_encoder=None, network=None, prompt_scripts_list=None, replace_word=None,\
                               embeddings_anchor_cache=None, scale=0.001, mixup=True):
        
        inds = []
        for idx_word in range(2*prompt_pair.sampling_batch_size * prompt_pair.target.shape[0]):
            inds.append(random.randint(0,embeddings_anchor_cache.size(0)-1))
        
        embs = embeddings_anchor_cache[inds]
                
        D,H,W = embs.shape[0], embs.shape[1], embs.shape[2]
             

        scale = 0.001
        noise = scale * embs.view(D, -1).norm(2, dim=1, keepdim=True).unsqueeze(-1) * torch.randn_like(embs)
        samples = embs + noise    
        
        samples_pair = samples.view(D//2, 2, H, W)
        
        #### MixUp #####
        mixRate = torch.tensor(np.random.beta(1.0, 1.0, (prompt_pair.sampling_batch_size * prompt_pair.target.shape[0],1,1))).to(samples_pair.device)
        
        samples = mixRate*samples_pair[:,0,:] + (1-mixRate)*samples_pair[:,1,:] if mixup else samples_pair[:,0,:]

        if prompt_pair.unconditional.shape[0] == 1:
            samples = [torch.cat([prompt_pair.unconditional, samples[idx].unsqueeze(0)]) for idx in range(samples.shape[0])]
            samples = torch.cat(samples, dim=0).float()
        else:
            samples = [torch.cat([prompt_pair.unconditional[0].unsqueeze(0), samples[idx].unsqueeze(0)]) for idx in range(samples.shape[0])]
            samples = torch.cat(samples, dim=0).float()
                        
        return samples        
    


def sample_arbitrary(anchors, tokenizer=None, text_encoder=None):

    # sample from gaussian distribution
    noise = torch.randn_like(anchors)
    # normalize the noise
    noise = noise / noise.view(-1).norm(dim=-1)
    # compute the similarity

    scale = torch.rand(anchors.shape[0])* 0.4 + 0.8
        
    sample = scale.unsqueeze(-1).unsqueeze(-1).to(anchors.device) * noise * anchors.view(-1).norm(dim=-1)
    
    return sample  



def sample_uniform(prompt_pair: PromptEmbedsPair, tokenizer=None, text_encoder=None):
    samples = []
    while len(samples) < prompt_pair.sampling_batch_size:
        # sample from gaussian distribution
        noise = torch.randn_like(prompt_pair.target)
        # normalize the noise
        noise = noise / noise.view(-1).norm(dim=-1)
        # compute the similarity
        sim = torch.cosine_similarity(prompt_pair.target.view(-1), noise.view(-1), dim=-1)
        # the possibility of accepting the sample = 1 - sim
        
        scale = random.random() * 0.4 + 0.8
        sample = scale * noise * prompt_pair.target.view(-1).norm(dim=-1)
        samples.append(sample)
    
    samples = [torch.cat([prompt_pair.unconditional, s]) for s in samples]
    samples = torch.cat(samples, dim=0)
    return samples  


def sample(prompt_pair: PromptEmbedsPair, tokenizer=None, text_encoder=None):
    samples = []
    while len(samples) < prompt_pair.sampling_batch_size:
        while True:
            # sample from gaussian distribution
            noise = torch.randn_like(prompt_pair.target)
            # normalize the noise
            noise = noise / noise.view(-1).norm(dim=-1)
            # compute the similarity
            sim = torch.cosine_similarity(prompt_pair.target.view(-1), noise.view(-1), dim=-1)
            # the possibility of accepting the sample = 1 - sim
            if random.random() < 1 - sim:
                break
        scale = random.random() * 0.4 + 0.8
        sample = scale * noise * prompt_pair.target.view(-1).norm(dim=-1)
        samples.append(sample)
    
    samples = [torch.cat([prompt_pair.unconditional, s]) for s in samples]
    samples = torch.cat(samples, dim=0)
    return samples  
