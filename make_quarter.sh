# 只保留 1/4 分辨率（4倍下采样）数据集
# 注意，由于用于计算的图像

input_folder=$1

if [ ! -d $input_folder ]; then
    echo "[MAKE QUAT] Input folder '${input_folder}' does not exist, please check."
fi

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
opt_ids=(0 1 2 3 4 5 6 7)

for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}

done