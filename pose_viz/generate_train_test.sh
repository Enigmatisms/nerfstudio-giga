folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")

for folder in ${folders[@]}; do
    python ./visualize_scene_ext.py -i ../../dataset/images_and_cams/$folder/
    python ./visualize_scene_ext.py -m -i ../../dataset/images_and_cams/$folder/
done