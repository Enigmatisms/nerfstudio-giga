ns-render \
    --load-config ./outputs/Science_IGEV/depth-nerfacto/science_new/config.yml \
    --traj filename --camera-path-filename ../dataset/images_and_cams/full/pose_align/ScienceSquare/output.json \
    --output-path renders/Science_IGEV/ --output_format images

# CUDA_VISIBLE_DEVICES=1 ns-render \
#     --load-config ./outputs/HaiyanHall_IGEV/depth-nerfacto/haiyanhall/config.yml \
#     --traj filename --camera-path-filename /home/lzw/dataset/pose_align/HaiyanHall/output.json \
#     --output-path renders/HaiyanHall_IGEV/ --output_format images

# CUDA_VISIBLE_DEVICES=1 python ./train.py