
# $1: 到对应训练脚本的路径(第一个训练)（不加.sh）
# $2: 到对应训练脚本的路径(第二个训练)
# $3: 到训练集所在的路径
# $4: 模式，如 _no_skew, 空 或者 _colmap
# $5: GPU device 号
echo "[TRAIN NOSKEW] $1 started to run..." 
CUDA_VISIBLE_DEVICES=$5 $1.sh $3 $4

echo "[TRAIN NOSKEW] $1 completed" 
echo "[TRAIN NOSKEW] $2 started to run..." 
CUDA_VISIBLE_DEVICES=$5 $2.sh $3 $4
echo "[TRAIN NOSKEW] $2 completed" 