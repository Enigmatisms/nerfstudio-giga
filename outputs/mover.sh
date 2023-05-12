folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")

for folder in ${folders[@]}; do
    path_in=${folder}_no_skew_1
    path_out=${folder}/depth-nerfacto/
    if [ ! -d $path_out ]; then
        mkdir -p $path_out
    fi
    mv $path_in $path_out
done