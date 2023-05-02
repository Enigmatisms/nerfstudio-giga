
# For path visualization for pose comparison, please use -v
# To see the result without post processing, set a big <--invalid_th>, like 100 and <--iteration> 1
# To generate output.json file for rendering, remove -v
# To generate original poses, use -o (skip pose registration and load from test.json)

scale=1
folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare")
# folders=("Library")

for folder in ${folders[@]}; do
    python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name $folder --output_name "output$1.json" \
        --colmap_name transforms$1.json \
        --scale $scale --invalid_th 100 -v
done

python3 ./colmap_icp.py --input_path ../../dataset/images_and_cams/full/ \
        --scene_name theOldGate --output_name "output$1.json" \
        --colmap_name transforms$1.json \
        --scale $scale --invalid_th 100 -v -t
