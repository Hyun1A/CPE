# Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate (ICLR 2025)

### ✏️ [Project Page](https://hyun1a.github.io/cpe.io) | 📄 [Paper](https://openreview.net/forum?id=ZRDhBwKs7l)

> **Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate**<br>
> Byung Hyun Lee, Sungjin Lim, Seunggyu Lee, Dong Un Kang, Se Young Chun <br>
> 
>**Abstract**: Remarkable progress in text-to-image diffusion models has brought a major concern about potentially generating images on inappropriate or trademarked concepts. Concept erasing has been investigated with the goals of deleting target concepts in diffusion models while preserving other concepts with minimal distortion. To achieve these goals, recent concept erasing methods usually fine-tune the cross-attention layers of diffusion models. In this work, we first show that merely updating the cross-attention layers in diffusion models, which is mathematically equivalent to adding \emph{linear} modules to weights, may not be able to preserve diverse remaining concepts. Then, we propose a novel framework, dubbed Concept Pinpoint Eraser (CPE), by adding \emph{nonlinear} Residual Attention Gates (ResAGs) that selectively erase (or cut) target concepts while safeguarding remaining concepts from broad distributions by employing an attention anchoring loss to prevent the forgetting. Moreover, we adversarially train CPE with ResAG and learnable text embeddings in an iterative manner to maximize erasing performance and enhance robustness against adversarial attacks. Extensive experiments on the erasure of celebrities, artistic styles, and explicit contents demonstrated that the proposed CPE outperforms prior arts by keeping diverse remaining concepts while deleting the target concepts with robustness against attack prompts.
<br>

![overview](./assets/fig1_v12.png)
(a) Comparison of fine-tuning approaches for concept erasing. Previous methods could affect both on target and remaining concepts as they merely fine-tunes CA layers. In contrast, our method, CPE, can adatively transmit the change for target concepts to erase while successfully suppressing it for remaining concepts, by using the proposed ResAGs. (b) Qualitative results on erasing “Claude Monet” artistic style, comparing with a baseline. 
<br>
<br>

![approach](assets/figure_main_cpe_v3.png)
(a) Architecture of ResAG module in CA layers for selectively erasing a target concept while preserving remaining concepts. (b) To erase multiple targets during inference, we merge multiple ResAGs by only adding the ResAG of the target with the highest gate value for each token. 
<br>
<br>


## Getting Started

### Setup for experiments

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


## Training and Sampling 
### Experiments: Celebrities Erasure

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


### Experiments: Artistic Styles Erasure

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
2. __We have already provided pre-trained ResAGs for 100 artistic styles in ./output, so you can execute the generation without training__


### Experiments: Explicit Contents Erasure

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


## Evaluation Metrics

In our paper, CPE, we utilize various metrics including [FID](https://github.com/GaParmar/clean-fid)(Fréchet Inception Distance), [KID](https://github.com/GaParmar/clean-fid)(Kernel Inception Distance), [CLIP score](https://github.com/openai/CLIP), [GIPHY Celebrity Detector](https://github.com/Giphy/celeb-detection-oss), and [NudeNet Detector](https://pypi.org/project/nudenet/) for explicit images.

**Evaluate FID / KID**
<pre>
sh ./shell_scripts/evaluation_metrics/evaluate_fid_kid.sh
</pre>

**Evaluate CLIP Score**
<pre>
sh ./shell_scripts/evaluation_metrics/evaluate_clip_score.sh
</pre>

**Evaluate GIPHY Celebrity Detector**

To use the GIPHY Celebrity Detector, download the official GCD code and create a conda environment for GCD by following the [official guide](https://github.com/Giphy/celeb-detection-oss). 
(Note that the GCD Python environment is not compatible with the CPE environment.) 
After setting up the GCD environment, please refer to [our installation guide](https://github.com/Hyun1A/CPE/tree/main/metrics) in the 'metrics' folder.

<pre>
sh ./shell_scripts/evaluation_metrics/evaluate_giphy_score.sh
</pre>

**Evaluate NudeNet Detector**
<pre>
sh ./shell_scripts/evaluation_metrics/evaluate_I2P.sh
</pre>


## Acknowledgements
We thank the following contributors that our code is based on: [SPM](https://github.com/Con6924/SPM?tab=readme-ov-file), [MACE](https://github.com/Shilin-LU/MACE?tab=readme-ov-file).

## Citation
If you find the repo useful, please consider citing.

<pre>
@InProceedings{lee2024cpe,
    author    = {Lee, Byung Hyun and Lim, Sungjin and Lee, Seunggyu and Kang, Dong Un and Chun, Se Young},
    title     = {Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate},
    booktitle = {ICLR},
    year      = {2025},
}
</pre>
