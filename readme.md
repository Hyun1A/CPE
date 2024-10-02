# Official Implementation of CPE (Concept Pinpoint Eraser for Text-to-image Diffusion Models via Residual Attention Gate)


## Getting Started
### Setup for experiments
Please install packages in requirements.txt
<pre>
pip install -r requirements.txt
</pre>

##### Celebrities Erasure #####
# Train:
sh ./shell_scripts/celebs/train_celeb_cpe_single.sh

# Generation:
sh ./shell_scripts/celebs/generate_by_celeb_model.sh

# 1. For generation of diverse domains such as artistic styles or characters from celebrity-erased model, 
#    please change the variable GEN_CONFIG in ./shell_scripts/celebs/generate_by_celeb_model.sh (config files for different domains are listed)
# 2. We have already provided pre-trained ResAGs for three celebrities in ./output, so you can execute the generation without training




### Artistic Styles Erasure ###
# Train:
sh ./shell_scripts/artists/train_artist_cpe_single.sh

# Generation:
sh ./shell_scripts/artists/generate_by_artist_model.sh

# 1. For generation of diverse domains such as celebrites or characters from celebrity-erased model, 
#    please change the variable GEN_CONFIG in ./shell_scripts/artists/generate_by_artist_model.sh (config files for different domains are listed)
# 2. We have already provided pre-trained ResAGs for three artistic styles in ./output, so you can execute the generation without training





## Explicit Contents Erasure ##
# Train:
sh ./shell_scripts/explicit/train_explicit_cpe_single.sh

# Generation (Explicit contents):
sh ./shell_scripts/explicit/generate_by_explicit_model_explicit.sh

# Generation (COCO-30K):
sh ./shell_scripts/explicit/generate_by_explicit_model_coco.sh

# 1. We have already provided pre-trained ResAGs for four explicit concepts in ./output, so you can execute the generation without training
