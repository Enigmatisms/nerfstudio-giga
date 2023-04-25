CUDA_VISIBLE_DEVICES=1 ns-render \
    --load-config ./outputs/Science_IGEV/depth-nerfacto/2023-04-23_165338/config.yml \
    --traj filename --camera-path-filename /home/lzw/dataset/pose_align/ScienceSquare/output.json \
    --output-path renders/Science_IGEV/ --output_format images

CUDA_VISIBLE_DEVICES=1 ns-render \
    --load-config ./outputs/theOldGate_IGEV/depth-nerfacto/2023-04-23_223329/config.yml \
    --traj filename --camera-path-filename /home/lzw/dataset/pose_align/theOldGate/output.json \
    --output-path renders/theOldGate_IGEV/ --output_format images