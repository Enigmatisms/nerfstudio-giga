data_set_path=$1

# 将 optimized_poses.json 中经过优化的位姿输出到 dataset 位置

echo "[OPT REPLACE] Pose replacer started."

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")  # 160 for newest config rendering in cuda 2
modes=("none" "colmap" "colmap" "colmap" "colmap" "colmap" "colmap" "colmap")

for ((i=0;i<7;i++)); do
    python ./pose_viz/replace_pose.py -s ${folders[$i]}_opt -n ${folders[$i]} \
            -m ${modes[$i]} -o $data_set_path --scale 8.0
done

echo "[OPT REPLACE] Pose replacer completed."