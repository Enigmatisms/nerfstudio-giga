# folders=("HaiyanHall")
folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
for folder in ${folders[@]}; do
    if [ ! -d ./to_zip/$folder ]; then
        mkdir -p ./to_zip/$folder
        # mkdir -p ./to_zip/${folder}_opt
    fi
    cp -r outputs/$folder/depth-nerfacto/${folder}_colmap_high ./to_zip/$folder
    # cp -r outputs/${folder}_opt/depth-nerfacto/${folder}_colmap_high ./to_zip/${folder}_opt
    # rm -r ./to_zip/${folder}_opt/${folder}_colmap_high/nerfstudio_models
done

zip -r zipped.zip ./to_zip