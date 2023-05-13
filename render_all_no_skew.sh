dataset_path=$1
folders=("theOldGate")

# # 四个进程用于渲染7个场景的 1/8 分辨率权限
# pids=(0 0 0)
# for ((i=0;i<3;i++)); do
#     idx1=$((2 * $i))
#     idx2=$((2 * $i + 1))
#     CUDA_VISIBLE_DEVICES=$i ./render_two.sh ${folders[$idx1]} ${folders[$idx2]} $dataset_path _no_skew $i &
#     pids[$i]=$!
# done

echo "[RENDER NOSKEW] no skew model rendering started."
for folder in ${folders[@]}; do
    CUDA_VISIBLE_DEVICES=0 ./render_one.sh ${folder} $dataset_path _no_skew
done

# for pid in ${pids[@]}; do
#     wait $pid
# done
echo "[RENDER NOSKEW] no skew model rendering completed."
