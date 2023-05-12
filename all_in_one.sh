#!/bin/bash
set -e
# 只要有 process 不成功，shell将会自动结束进程，不会继续执行

# FIXME: 所有 shell 脚本的参数没有设置

# # 生成一倍降采样图像用于 HLOC 计算位姿与内参
# echo "[INFO] ffmpeg_scale.sh: Scaling images..."
# # $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）主要使用 images/ 文件夹
# ./ffmpeg_scale.sh $1

# # 根据 cam.txt 生成训练 json
# # QUITE
# echo "[INFO] generate_train_test.sh: Generating json files..."
# # $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）主要使用 cam/ 文件夹
# ./generate_train_test.sh $1

# # HLOC 位姿计算
# echo "[INFO] hloc_pose_gen.sh: HLOC, two processes, pose refinement..."

# if [ ! -d $2 ]; then
#     echo "[INFO] '$2' does not exist, creating folder..."
#     mkdir -p $2
# fi

# # [UNIT TEST-1]: 生成原始数据集
# 两个进程跑 HLOC 位姿计算，考虑到有多 GPU，但尚不清楚 CPU/内存的占用 --- TODO: 此处或许可以激进一些
# $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）
# $2: HLOC 的输出位置，我们将在此处建立真正的训练数据集

# Version 1: 两进程
# ./hloc_pose_gen1.sh $1 $2  &
# pid=$!
# ./hloc_pose_gen2.sh $1 $2
# wait $pid

# Version 2: 四进程
# pids=(0 0 0 0)
# folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden")
# for ((i=0;i<3;i++)); do
#     idx1=$((2 * $i))
#     idx2=$((2 * $i + 1))
#     ./hloc_pose_gen.sh $1 $2 ${folders[$idx1]} ${folders[$idx2]} $i &
#     pids[$i]=$!
# done
# ./hloc_pose_gen.sh $1 $2 ScienceSquare theOldGate 3
# for pid in ${pids[@]}; do
#     wait $pid
# done

# Version (测试用): 单进程
./hloc_pose_gen1.sh $1 $2
./hloc_pose_gen2.sh $1 $2
echo "[INFO] HLOC completed."

echo "[INFO] hloc_pose_replace.sh: HLOC poses are being replaced..."

# HLOC 位姿替换
# $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）
# $2: HLOC 的输出位置，从 $1 读取的 train.json / test.json 在此处复制
./hloc_pose_replace.sh $1 $2
echo "[INFO] HLOC pose replacing completed."

# [TODO] [UNIT_TEST-2 生成 HLOC 数据集]
# HLOC 将会生成 4种分辨率的图片，我们只留1/4分辨率图像，TODO: 首先需要知道前面所有的流程生成出来是什么样的