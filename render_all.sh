dataset_path=$1
folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
modes=("" "_colmap" "_colmap" "_colmap" "_colmap" "_colmap" "_colmap" "_colmap")

# pids=(0 0 0)
# for ((i=0;i<3;i++)); do
#     idx1=$((2 * $i))
#     idx2=$((2 * $i + 1))
#     CUDA_VISIBLE_DEVICES=$i ./render_two.sh ${folders[$idx1]} ${folders[$idx2]} $dataset_path _no_skew $i &
#     pids[$i]=$!
# done

echo "[RENDER MODEL] full model rendering started."
for ((i=0;i<8;i++)); do
    folder=${folders[$i]}
    mode=${modes[$i]}
    # 需要使用 _opt 作为后缀，因为我们使用进行了位姿 + 内参优化的结果进行渲染
    CUDA_VISIBLE_DEVICES=0 ./render_one.sh ${folder} $dataset_path $mode _opt
done

# for pid in ${pids[@]}; do
#     wait $pid
# done
echo "[RENDER MODEL] full model rendering completed."
