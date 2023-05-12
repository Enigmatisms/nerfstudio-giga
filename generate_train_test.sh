
# 为七个场景生成融合了训练/测试的 json 文件
folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")

echo "[DATA GEN] Dataset generation path is ${1}"

if [ ! -d $1 ]; then
    echo "[DATA GEN] Dataset path does not exist, please check."
fi

# -m 表示 --merge
for folder in ${folders[@]}; do
    python3 ./pose_viz/visualize_scene_ext.py -m -i ${1}$folder/ -n $folder
done

# DayaTemple 则无需生成包含测试位姿的 json 文件
python3 ./pose_viz/visualize_scene_ext.py -i ${1}DayaTemple/ -n DayaTemple

echo "[DATA GEN] Dataset generation completed."
