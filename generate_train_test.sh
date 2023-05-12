
# 为七个场景生成融合了训练/测试的 json 文件
folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")

input_folder=$1
cp_output_folder=$2
echo "[DATA GEN] Dataset generation path is ${input_folder}"

if [ ! -d $input_folder ]; then
    echo "[DATA GEN] Dataset path does not exist, please check."
fi

# -m 表示 --merge
for folder in ${folders[@]}; do
    # -m 与 no_merge 都需要生成一份（以供 HLOC 使用）
    input_path=${input_folder}$folder/
    python3 ./pose_viz/visualize_scene_ext.py -m -i $input_path -n $folder
    python3 ./pose_viz/visualize_scene_ext.py -i $input_path -n $folder
    cp ${input_path}train*.json $cp_output_folder$folder/
    cp ${input_path}test.json $cp_output_folder$folder/
done

# DayaTemple 则无需生成包含测试位姿的 json 文件
python3 ./pose_viz/visualize_scene_ext.py -i ${input_folder}DayaTemple/ -n DayaTemple
cp ${input_folder}DayaTemple/train*.json ${cp_output_folder}DayaTemple/
cp ${input_folder}DayaTemple/test.json ${cp_output_folder}DayaTemple/

echo "[DATA GEN] Dataset generation completed."
