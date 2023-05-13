data_set_path=$1

echo "[TRAIN MODEL] Training started." 

files=("daya_temple" "haiyan_hall" "library" "memorial_hall" "museum" "peony_garden" "science_square" "the_old_gate")
opt_ids=(0 1 2 3 4 5 6 7)
for idx in ${opt_ids[@]}; do
    file=${files[$idx]}
    CUDA_VISIBLE_DEVICES=0 ./configs/$file.sh $data_set_path
done

echo "[TRAIN MODEL] Training completed, prepare to optimize poses" 
