#!/bin/bash
set -e
# 只要有 process 不成功，shell将会自动结束进程，不会继续执行

# FIXME: 所有 shell 脚本的参数没有设置

# 生成一倍降采样图像用于 HLOC 计算位姿与内参
# echo "[INFO] ffmpeg_scale.sh: Scaling images..."
# # $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）主要使用 images/ 文件夹
# ./ffmpeg_scale.sh $1

# 根据 cam.txt 生成训练 json
# TODO: QUITE
echo "[INFO] generate_train_test.sh: Generating json files..."
# $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）主要使用 cam/ 文件夹
# $2: 我们需要将 train/test.json 先行复制到 dataset 输出文件夹中
./generate_train_test.sh $1 $2
echo "[INFO] generate_dinger.sh: Generating IGEV/masking format json..."
./generate_dinger.sh $1 $2
# HLOC 位姿计算
echo "[INFO] hloc_pose_gen.sh: HLOC, two processes, pose refinement..."

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

if [ ! -d $2 ]; then
    echo "[INFO] '$2' does not exist, creating folder..."
    mkdir -p $2
fi

# HLOC 将会生成 4种分辨率的图片，我们只留1/4分辨率图像
# Version (测试用): 单进程
# ./hloc_pose_gen1.sh $1 $2
# ./hloc_pose_gen2.sh $1 $2
echo "[INFO] HLOC completed."

# echo "[INFO] hloc_pose_replace.sh: HLOC poses are being replaced..."

# HLOC 位姿替换
# $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）
# $2: HLOC 的输出位置，从 $1 读取的 train.json / test.json 在此处复制
./hloc_pose_replace.sh $1 $2
echo "[INFO] HLOC pose replacing completed."

# [UNIT_TEST-2 生成 HLOC 数据集]
echo "[INFO] HLOC dataset without depth completed"

# 基本数据集的生成到此结束 --- IGEV 进一步处理数据集将在外部进行


# TODO: datagen 还有需要修改的地方: 生成 IGEV/sky-mask/reprojection-inpainting 所需要的结果

# data_gen.sh 到 train_phase 还差的环节
# (1) HLOC 的调整
# (2) IGEV 深度图 与 sky_masking 的生成
# train_phase 中的重要一步： 重投影对应的 inpainting
# 另外：verbose 等级 / 多卡训练需要开启