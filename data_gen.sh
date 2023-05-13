#!/bin/bash
set -e
# 只要有 process 不成功，shell将会自动结束进程，不会继续执行

# FIXME: 所有 shell 脚本的参数没有设置

# 生成一倍降采样图像用于 HLOC 计算位姿与内参
# echo "[INFO] ffmpeg_scale.sh: Scaling images..."
# # $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）主要使用 images/ 文件夹
# ./ffmpeg_scale.sh $1

# # 根据 cam.txt 生成训练 json
# # TODO: QUITE
# echo "[INFO] generate_train_test.sh: Generating json files..."
# # $1: 原始数据集的位置（此目录下应该有 DayaTemple, HaiyanHall, ...）主要使用 cam/ 文件夹
# # $2: 我们需要将 train/test.json 先行复制到 dataset 输出文件夹中
# ./generate_train_test.sh $1 $2

# # HLOC 位姿计算
# echo "[INFO] hloc_pose_gen.sh: HLOC, two processes, pose refinement..."

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

# if [ ! -d $2 ]; then
#     echo "[INFO] '$2' does not exist, creating folder..."
#     mkdir -p $2
# fi

# HLOC 将会生成 4种分辨率的图片，我们只留1/4分辨率图像
# Version (测试用): 单进程
# ./hloc_pose_gen1.sh $1 $2
./hloc_pose_gen2.sh $1 $2
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

# 此后的步骤
# 假设我们已经有了各种训练所需的内容，首先就是要完成训练
# 第一个需要完成的训练是7个no_skew场景的训练
# 训练结束后，使用 run_icp.sh 生成对应的 output_no_skew.json 进行渲染（只渲染七个场景）
# 渲染结束后保存在renders中，此时 DayaTemple还在 other_data 中
# 注意测试集优化的位姿不是 no_skew: DayaTemple 是 DayaTemple 模型 output.json 转的
# 其他场景则是其他的 output_colmap.json 转的 （最终成为 transforms_colmap_opt.json）

# 完成训练之后，可以使用 run_icp.sh 首先生成对应的 output.json 结果
# 例如 DayaTemple生成 output.json, 其余为 output_colmap.json
# 拷贝 dataparser_transforms.json，需要两次？个人觉得，不要拷贝，使用一个参数喂入，拷贝太麻烦了
# 使用这些 json + dataparser_transforms.json 生成位姿优化数据集
# DayaTemple需要单独拷贝，其他都是直接拷贝自 renders
# camera_opt 负责拷贝，速度应该较快。此外，根据对应的 outputxxxx.json 生成位姿配准数据集
# 生成完数据集即可开始进行位姿配准位姿配准可以使用之前的model进行测试