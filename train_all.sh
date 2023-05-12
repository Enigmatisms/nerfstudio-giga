data_set_path=$1

echo "[TRAIN MODEL] Training started." 

files=("daya_temple" "haiyan_hall" "library" "memorial_hall" "museum" "peony_garden" "science_square")
no_colmap=(1 0 0 0 0 0 0 0)
opt_ids=(0 1 2 3 4 5 6 7)
for idx in ${opt_ids[@]}; do
    file=${files[$idx]}
    no_colmap_flag=${no_colmap[$idx]}
    if [ $no_colmap_flag -eq 0 ]; then
        CUDA_VISIBLE_DEVICES=0 ./configs/$files.sh $data_set_path _colmap
    else
        CUDA_VISIBLE_DEVICES=0 ./configs/$files.sh $data_set_path
    fi
done

echo "[TRAIN MODEL] Training completed, prepare to optimize poses" 
