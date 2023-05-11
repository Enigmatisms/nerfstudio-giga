folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")

for folder in ${folders[@]}; do
    cp -r ./renders/$folder/ ../dataset/${folder}_opt/images
done
