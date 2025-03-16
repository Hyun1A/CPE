

CONFIG="./configs/train_celeb/config.yaml"
GATE_RANK="16"
GUIDE="0.3"
PAL="100000.0"
ST_IDX="0"
END_IDX="50"
NOISE="0.001"


python ./train/train_cpe.py \
    --config_file ${CONFIG} \
    --st_prompt_idx ${ST_IDX} \
    --end_prompt_idx ${END_IDX} \
    --gate_rank ${GATE_RANK} \
    --guidance_scale ${GUIDE} \
    --pal ${PAL} \
    --noise ${NOISE}
    