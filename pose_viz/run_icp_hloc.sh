
# For path visualization for pose comparison, please use -v
# To see the result without post processing, set a big <--invalid_th>, like 100 and <--iteration> 1
# To generate output.json file for rendering, remove -v
# To generate original poses, use -o (skip pose registration and load from test.json)

scale=1

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name DayaTemple --output_name "output.json" \
        --colmap_name transforms.json \
        --scale $scale --invalid_th 1

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name HaiyanHall --output_name "output.json" \
        --colmap_name transforms.json \
        --scale $scale --invalid_th 100

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name Library --output_name "output.json" \
        --colmap_name transforms.json \
        --scale $scale --invalid_th 100

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name MemorialHall --output_name "output.json" \
        --colmap_name transforms.json \
        --scale $scale --invalid_th 100

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name Museum --output_name "output.json" \
        --colmap_name transforms.json \
        --scale $scale --invalid_th 100

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name PeonyGarden --output_name "output.json" \
        --colmap_name transforms.json \
        --scale $scale --invalid_th 100

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name ScienceSquare --output_name "output.json" \
        --colmap_name transforms.json \
        --scale $scale --invalid_th 100

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name theOldGate --output_name "output.json" \
        --colmap_name transforms.json \
        --scale $scale --invalid_th 100 -t
