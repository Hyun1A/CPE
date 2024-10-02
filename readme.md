<p align="center" style="font-size:50px;">
# Official Implementation of CPE 
## (Concept Pinpoint Eraser via Residual Attention Gate)
</p>

## Getting Started
### Setup for experiments
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
2. We have already provided pre-trained ResAGs for three celebrities in ./output, so you can execute the generation without training


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
2. We have already provided pre-trained ResAGs for three artistic styles in ./output, so you can execute the generation without training


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
