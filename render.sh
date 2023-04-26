CUDA_VISIBLE_DEVICES=1 ns-render \
    --load-config ./outputs/MemorialHall_IGEV/depth-nerfacto/2023-04-23_163535/config.yml \
    --traj filename --camera-path-filename /home/lzw/dataset/pose_align/MemorialHall/output.json \
    --output-path renders/MemorialHall_IGEV/ --output_format images

CUDA_VISIBLE_DEVICES=1 ns-render \
    --load-config ./outputs/HaiyanHall_IGEV/depth-nerfacto/haiyanhall/config.yml \
    --traj filename --camera-path-filename /home/lzw/dataset/pose_align/HaiyanHall/output.json \
    --output-path renders/HaiyanHall_IGEV/ --output_format images

CUDA_VISIBLE_DEVICES=1 python ./train.py