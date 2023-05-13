
# For path visualization for pose comparison, please use -v
# To see the result without post processing, set a big <--invalid_th>, like 100 and <--iteration> 1
# To generate output.json file for rendering, remove -v
# To generate original poses, use -o (skip pose registration and load from test.json)

dataset_path=$1
mode=$2
scale=$3
process_daya=$4

# scale: 对于测试集位姿优化而言，为0.125, mode 则为 _no_skew, $4 = 0
# 对于实际的渲染生成outputxxxx.json 而言，scale = 1.0, mode = _colmap, $4 = 1

folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare")

for folder in ${folders[@]}; do
    python3 ./pose_viz/colmap_icp.py --input_path $dataset_path \
        --scene_name $folder --output_name "output${mode}.json" \
        --colmap_name transforms${mode}.json \
        --parser_name ./outputs/$folder/depth-nerfacto/${folder}${mode}/dataparser_transforms.json \
        --scale $scale --invalid_th 100 -o
done

# theOldGate 单独处理
python3 ./pose_viz/colmap_icp.py --input_path $dataset_path \
        --scene_name theOldGate --output_name "output${mode}.json" \
        --colmap_name transforms${mode}.json \
        --parser_name ./outputs/theOldGate/depth-nerfacto/theOldGate${mode}/dataparser_transforms.json \
        --scale $scale --invalid_th 100 -o -t

# DayaTemple无需渲染训练位姿优化图，故只需要一次
if [ $process_daya -gt 0 ]; then
    echo "Processing DayaTemple."
    python3 ./pose_viz/colmap_icp.py --input_path $dataset_path \
        --scene_name DayaTemple --output_name "output.json" \
        --colmap_name transforms.json \
        --parser_name ./outputs/DayaTemple/depth-nerfacto/DayaTemple/dataparser_transforms.json \
        --scale $scale --invalid_th 100 --icp_th 2 --decay 0.4 -o
fi
