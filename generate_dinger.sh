raw_data_path=$1
data_set_path=$2

# 生成 IGEV / sky mask 所需json 文件

echo "[DINGER GEN] Dataset generation dinger format"

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")  # 160 for newest config rendering in cuda 2

for ((i=0;i<8;i++)); do
    python ./pose_viz/generate_dinger.py ${raw_data_path}${folders[$i]}/
    cp ${raw_data_path}${folders[$i]}/transforms_dinger.json ${data_set_path}${folders[$i]}/
    cp ${raw_data_path}${folders[$i]}/test_dinger.json ${data_set_path}${folders[$i]}/
done

echo "[DINGER GEN] Dataset generation dinger format completed"