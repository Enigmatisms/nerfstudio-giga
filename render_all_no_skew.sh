dataset_path=$1

# 四个进程用于渲染7个场景的 1/8 分辨率权限
pids=(0 0 0)
folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "theOldGate")
for ((i=0;i<3;i++)); do
    idx1=$((2 * $i))
    idx2=$((2 * $i + 1))
    CUDA_VISIBLE_DEVICES=$i ./render_two.sh ${folders[$idx1]} ${folders[$idx2]} $dataset_path _no_skew $i &
    pids[$i]=$!
done

echo "[TRAIN NOSKEW] ./config_no_skew/the_old_gate started to run..." 
CUDA_VISIBLE_DEVICES=3 ./render_two.sh ${folders[$idx1]} ${folders[$idx2]} $dataset_path _no_skew 3
echo "[TRAIN NOSKEW] ./config_no_skew/the_old_gate completed" 

for pid in ${pids[@]}; do
    wait $pid
done
echo "[TRAIN NOSKEW] no skew model training completed, prepare to render 1/8 resolution images..."
