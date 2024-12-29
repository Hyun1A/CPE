# Official Implementation of "Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate"

### ✏️ [Project Page](https://hyun1a.github.io/cpe.io) | 📄 [Paper]

Official Implementation of CPE 

> **MACE: Mass Concept Erasure in Diffusion Models**<br>
<!-- > [Gwanghyun Kim](https://gwang-kim.github.io/), Taesung Kwon, [Jong Chul Ye](https://bispl.weebly.com/professor.html) <br> -->
> Shilin Lu, Zilan Wang, Leyang Li, Yanzhu Liu, Adams Wai-Kin Kong <br>
> CVPR 2024
> 
>**Abstract**: <br>
The rapid expansion of large-scale text-to-image diffusion models has raised growing concerns regarding their potential misuse in creating harmful or misleading content. In this paper, we introduce MACE, a finetuning framework for the task of mass concept erasure. This task aims to prevent models from generating images that embody unwanted concepts when prompted. Existing concept erasure methods are typically restricted to handling fewer than five concepts simultaneously and struggle to find a balance between erasing concept synonyms (generality) and maintaining unrelated concepts (specificity). In contrast, MACE differs by successfully scaling the erasure scope up to 100 concepts and by achieving an effective balance between generality and specificity. This is achieved by leveraging closed-form cross-attention refinement along with LoRA finetuning, collectively eliminating the information of undesirable concepts. Furthermore, MACE integrates multiple LoRAs without mutual interference. We conduct extensive evaluations of MACE against prior methods across four different tasks: object erasure, celebrity erasure, explicit content erasure, and artistic style erasure. Our results reveal that MACE surpasses prior methods in all evaluated tasks.











## Setup for experiments

**OS**: Ubuntu 18.04.5 LTS

**Python**: 3.9.19
<pre>
conda create -n CPE python=3.9
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install xformers
</pre>

Please install packages in requirements.txt
<pre>
pip install -r requirements.txt
</pre>

## Running Experiments
### Celebrities Erasure
**Train:**
<pre>
sh ./shell_scripts/celebs/train_celeb_cpe_single.sh
</pre>

**Generation:**
<pre>
sh ./shell_scripts/celebs/generate_by_celeb_model.sh
</pre>

1. For generation of diverse domains such as artistic styles or characters from celebrity-erased model, please change the variable GEN_CONFIG in
   ./shell_scripts/celebs/generate_by_celeb_model.sh (config files for different domains are listed)
2. We have already provided pre-trained ResAGs for 50 celebrities in ./output, so you can execute the generation without training


### Artistic Styles Erasure
**Train:**
<pre>
sh ./shell_scripts/artists/train_artist_cpe_single.sh
</pre>

**Generation:**
<pre>
sh ./shell_scripts/artists/generate_by_artist_model.sh
</pre>

1. For generation of diverse domains such as celebrites or characters from celebrity-erased model, 
   please change the variable GEN_CONFIG in ./shell_scripts/artists/generate_by_artist_model.sh (config files for different domains are listed)
2. We have already provided pre-trained ResAGs for 100 artistic styles in ./output, so you can execute the generation without training


## Explicit Contents Erasure
**Train:**
<pre>
sh ./shell_scripts/explicit/train_explicit_cpe_single.sh
</pre>

**Generation (Explicit contents):**
<pre>
sh ./shell_scripts/explicit/generate_by_explicit_model_explicit.sh
</pre>

**Generation (COCO-30K):**
<pre>
sh ./shell_scripts/explicit/generate_by_explicit_model_coco.sh
</pre>

1. We have already provided pre-trained ResAGs for four explicit concepts in ./output, so you can execute the generation without training
