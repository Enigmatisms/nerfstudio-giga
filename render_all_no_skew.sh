dataset_path=$1
folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
image_nums=(25 53 17 54 42 32 47)
# # 四个进程用于渲染7个场景的 1/8 分辨率权限
# pids=(0 0 0)
# for ((i=0;i<3;i++)); do
#     idx1=$((2 * $i))
#     idx2=$((2 * $i + 1))
#     CUDA_VISIBLE_DEVICES=$i ./render_two.sh ${folders[$idx1]} ${folders[$idx2]} $dataset_path _no_skew $i &
#     pids[$i]=$!
# done

echo "[RENDER NOSKEW] no skew model rendering started."
for ((i=0;i<7;i++)); do
    folder=${folders[$i]}
    img_num=${image_nums[$i]}
    CUDA_VISIBLE_DEVICES=1 ./render_one.sh ${folder} $dataset_path _no_skew 0.1 128 $img_num
done

# for pid in ${pids[@]}; do
#     wait $pid
# done
echo "[RENDER NOSKEW] no skew model rendering completed."
