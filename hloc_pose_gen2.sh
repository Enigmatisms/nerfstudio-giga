input_folder=$1
output_folder=$2

if [ ! -d $input_folder ]; then
    echo "[HLOC POSE - 2] Input folder '${input_folder}' does not exist, please check."
fi

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
opt_ids=(4 5 6 7)
for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}
    CUDA_VISIBLE_DEVICES=0 ns-process-data images \
        --data $input_folder$folder/images_scaled/ \
        --output-dir $output_folder$folder/ \
        --sfm-tool hloc --refine-pixsfm --use-sfm-depth

    mod_path=${output_folder}${folder}/
    to_rm=("images_8" "colmap" "depth" "depths_2" "depths_4" "depths_8")
    # 删除所有 depths 文件夹
    for target in ${to_rm[@]}; do
        if [ -d ${mod_path}${target} ]; then
            rm -r ${mod_path}${target}
        fi
    done
    if [ ! -d ${mod_path}depths/ ]; then
        mkdir -p ${mod_path}depths/
    fi
    if [ -d ${mod_path}images_2 ]; then
        rm -r ${mod_path}images
        mv ${mod_path}images_2 ${mod_path}images
    fi
done

echo "[HLOC POSE - 2] HLOC pose generation process 2 completed."