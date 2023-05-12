# 只保留 1/4 分辨率（4倍下采样）数据集
# 注意，由于用于计算的图像
# input_folder 是 HLOC 的输出路径

input_folder=$1

if [ ! -d $input_folder ]; then
    echo "[MAKE QUAT] Input folder '${input_folder}' does not exist, please check."
fi

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
opt_ids=(0 1 2 3 4 5 6 7)

for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}
    mod_path=${input_folder}${folder}/
    to_rm=("images_8" "colmap" "depths" "depths_2" "depths_4" "depths_8")
    # 删除所有 depths 文件夹
    for target in ${to_rm[@]}; do
        if [ -d ${mod_path}${target} ]; then
            rm -r ${mod_path}${target}
        fi
    done
    rm -r ${mod_path}depths/
    if [ -d ${mod_path}images_2 ];
        rm -r ${mod_path}images
        mv ${mod_path}images_2 ${mod_path}images
    fi
    # images_4 会被保留，用于计算IGEV深度图
done

echo "[MAKE_QUAT] Dataset cleaning up completed, there should be only one image folder in each scene folder."