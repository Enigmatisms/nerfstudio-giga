folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
names=("daya" "haiyan" "library" "memo" "museum" "peony" "science" "old_gate")

output_folder=./parser${1}
if [ ! -d $output_folder ]; then
    mkdir $output_folder
fi

for ((i=0;i<8;i++)); do
    path=${folders[$i]}/depth-nerfacto/${names[$i]}${1}/dataparser_transforms.json
    if [ ! -f $path ]; then
        echo "'$path' does not exist, skipping..."
    else
        output_path=$output_folder/${folders[$i]}
        if [ ! -d $output_path ]; then
            mkdir -p $output_path
        fi
        cp $path $output_path
    fi
done