################################################################
############## options for domains to generate #################

### celeb_erased: configs/gen_celeb/config_celeb_erased.yaml
### celeb_remained: configs/gen_celeb/config_celeb_remained.yaml
### artist_remained: configs/gen_celeb/config_artist.yaml
### character_remained: configs/gen_celeb/config_character.yaml
### coco_30K: configs/gen_celeb/config_coco_30k.yaml

############## options for domains to generate #################
################################################################

GEN_CONFIG=configs/gen_celeb/config_celeb_erased.yaml
GEN_ST_IDX=0
GEN_END_IDX=100000000
GATE_RANK="16"
GUIDE="0.3"
PAL="100000.0"
NOISE="0.001"

python ./generate/generate_cpe.py --config ${GEN_CONFIG} \
    --model_path "./output/Singleton_Celeb/celeb_50/celeb_single_guide"${GUIDE}"_pal"${PAL}"_gate_rank"${GATE_RANK}"_noise"${NOISE} \
    --save_env "guide"${GUIDE}"_pal"${PAL}"_gate_rank"${GATE_RANK}"_noise"${NOISE} \
    --st_prompt_idx ${GEN_ST_IDX} \
    --end_prompt_idx ${GEN_END_IDX} \
    --gate_rank ${GATE_RANK}
