folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")

for ((i=0;i<7;i++)); do
    path_in=${folders[$i]}_opt/depth-nerfacto/${folders[$i]}${1}
    path_out=${path_in}_bf
    mv $path_in $path_out 
done