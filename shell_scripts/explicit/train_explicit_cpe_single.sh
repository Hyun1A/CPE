

CONFIG="./configs/train_explicit/config.yaml"
GATE_RANK="64"
GUIDE="3.0"
PAL="10000.0"
ST_IDX="0"
END_IDX="4"
RESUME_STAGE="0"
NOISE="0.001"

python ./train/train_cpe.py \
    --config_file ${CONFIG} \
    --st_prompt_idx ${ST_IDX} \
    --end_prompt_idx ${END_IDX} \
    --gate_rank ${GATE_RANK} \
    --guidance_scale ${GUIDE} \
    --pal ${PAL} \
    --resume_stage ${RESUME_STAGE} \
    --noise ${NOISE}