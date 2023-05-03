folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")

for folder in ${folders[@]}; do
    python3 ./camera_opt.py -i $folder -m none -f
    python3 ./camera_opt.py -i $folder -m colmap -f
done