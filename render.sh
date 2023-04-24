CUDA_VISIBLE_DEVICES=1 ns-render \
    --load-config ./outputs/HaiyanHall_IGEV/depth-nerfacto/2023-04-23_152958/config.yml \
    --traj filename --camera-path-filename ../dataset/HaiyanHall_IGEV/output.json \
    --output-path renders/HaiyanHall_IGEV/ --output_format images