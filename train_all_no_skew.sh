dataset_path=$1

# 四个进程用于训练七个场景的 原始位姿 + 原始内参 版本
pids=(0 0 0)
files=("haiyan_hall" "library" "memorial_hall" "museum" "peony_garden" "science_square" )
for ((i=0;i<3;i++)); do
    idx1=$((2 * $i))
    idx2=$((2 * $i + 1))
    ./train_two.sh ${files[$idx1]} ${files[$idx2]} $dataset_path _no_skew $i &
    pids[$i]=$!
done
CUDA_VISIBLE_DEVICES=3 ./configs/the_old_gate.json $dataset_path _no_skew 3

for pid in ${pids[@]}; do
    wait $pid
done
