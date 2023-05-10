# inner_name=("museum_opt1" "peony_opt1") # "memo_opt" "haiyan_opt" "museum_opt" "peony_opt" "science_opt" "theoldgate_opt" "daya_opt")

folders=("theOldGate" "MemorialHall" "HaiyanHall" "Museum" "PeonyGarden" "ScienceSquare" "Library" "DayaTemple")  # 160 for newest config rendering in cuda 2
# inner_name=("theoldgate_opt1") # "memo_opt" "haiyan_opt" "museum_opt" "peony_opt" "science_opt" "theoldgate_opt" "daya_opt")

mode="colmap"

for ((i=0;i<7;i++)); do
    python ./replace_pose.py -s ${folders[$i]}_opt -n ${folders[$i]}_colmap_high -m $mode -o ../../dataset/ --scale 8.0
done
# python ./replace_pose.py -s DayaTemple_opt -n DayaTemple_high -m "none" -o ../../dataset/