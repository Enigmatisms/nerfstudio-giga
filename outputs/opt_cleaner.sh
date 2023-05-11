folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
small_folders=("haiyan" "library" "memo" "museum" "peony" "science" "old_gate")

for ((i=0;i<7;i++)); do
    # path_in=${folders[$i]}_opt/depth-nerfacto/${folders[$i]}${1}
    # if [ -d ${path_in}/nerfstudio_models ]; then
    #     rm -r ${path_in}/nerfstudio_models
    # fi

    path_in=${folders[$i]}/depth-nerfacto/${small_folders[$i]}${1}
    if [ -d ${path_in} ]; then
        rm -r ${path_in}
    fi
done