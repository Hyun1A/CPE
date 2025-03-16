
GEN_CONFIG=configs/gen_explicit/config_coco_30k.yaml
GEN_ST_IDX=0
GEN_END_IDX=100000000
GATE_RANK="64"
GUIDE="3.0"
PAL="10000.0"
NOISE="0.001"

python ./generate/generate_cpe.py --config ${GEN_CONFIG} \
    --model_path "./output/CPE_Explicit/explicit_4/explicit_single_guide"${GUIDE}"_pal"${PAL}"_gate_rank"${GATE_RANK}"_noise"${NOISE} \
    --save_env "guide"${GUIDE}"_pal"${PAL}"_gate_rank"${GATE_RANK}"_noise"${NOISE} \
    --st_prompt_idx ${GEN_ST_IDX} \
    --end_prompt_idx ${GEN_END_IDX} \
    --gate_rank ${GATE_RANK}