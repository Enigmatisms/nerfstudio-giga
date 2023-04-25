
# For path visualization for pose comparison, please use -v
# To see the result without post processing, set a big <--invalid_th>, like 100 and <--iteration> 1
# To generate output.json file for rendering, remove -v

# python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/pose_align/ \
#     --scene_name HaiyanHall \
#     --scale 1 --invalid_th 100

# python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/pose_align/ \
#     --scene_name MemorialHall \
#     --scale 1 --invalid_th 100

# python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/pose_align/ \
#     --scene_name ScienceSquare \
#     --scale 1 --invalid_th 100

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/pose_align/ \
    --scene_name theOldGate \
    --scale 1 --invalid_th 100