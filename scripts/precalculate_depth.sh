export MASTER_PORT=12000
export WORLD_SIZE=2
export MASTER_ADDR="127.0.0.1"

torchrun \
    --master-port=$MASTER_PORT \
    --nnodes=1 \
    --nproc_per_node=$WORLD_SIZE \
    -m data_scripts.get_stereo_depth