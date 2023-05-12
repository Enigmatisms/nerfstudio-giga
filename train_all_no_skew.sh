dataset_path=$1
files=("haiyan_hall" "library" "memorial_hall" "museum" "peony_garden" "science_square")

## 两个进程用于训练七个场景的 原始位姿 + 原始内参 版本 (多卡可用)
# pids=(0 0 0)
# for ((i=0;i<3;i++)); do
#     idx1=$((2 * $i))
#     idx2=$((2 * $i + 1))
#     ./train_two.sh ./config_no_skew/${files[$idx1]} ./config_no_skew/${files[$idx2]} $dataset_path _no_skew $i &
#     pids[$i]=$!
# done

for scene in ${files[@]}; do
    CUDA_VISIBLE_DEVICES=0 ./config_no_skew/$scene.sh $dataset_path _no_skew
done

echo "[TRAIN NOSKEW] ./config_no_skew/the_old_gate started to run..." 
CUDA_VISIBLE_DEVICES=0 ./config_no_skew/the_old_gate.sh $dataset_path _no_skew
echo "[TRAIN NOSKEW] ./config_no_skew/the_old_gate completed" 

# for pid in ${pids[@]}; do
#     wait $pid
# done
echo "[TRAIN NOSKEW] no skew model training completed, prepare to render 1/8 resolution images..."
