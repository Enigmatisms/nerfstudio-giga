input_folder=$1

if [ ! -d $input_folder ]; then
    echo "[IMG SCALE] Input folder '${input_folder}' does not exist, please check."
fi

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
transpose=(0 0 0 0 0 0 0 1)
# opt_ids=(0 1 2 3 4 5 6 7)
opt_ids=(7)
for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}
    scale_folder=$input_folder$folder/images_scaled
    if [ ! -d ${scale_folder} ]; then
        mkdir -p ${scale_folder}
    fi
    for file in `ls $input_folder$folder/images/*.jpg`; do
        name=`basename $file`
        if [ $name = "00000058.jpg" ] && [ $idx -eq 2 ]; then 
            continue
        fi
        out_file=${scale_folder}/$name
        if [ ${transpose[$idx]} -gt 0 ]; then       # theOldGate 需要 transpose
            ffmpeg -y -noautorotate -hide_banner -loglevel error -i $file -q:v 2 -vf "transpose=2, scale=iw/2:ih/2" $out_file
        else
            ffmpeg -y -noautorotate -hide_banner -loglevel error -i $file -q:v 2 -vf "scale=iw/2:ih/2" $out_file
        fi
    done
    echo "[IMG SCALE] Scene $folder processed, downscale factor 2"
done