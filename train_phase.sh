#!/bin/bash
# 此处开始我们真正考虑模型训练，此前都是进行数据集生成

dataset_path=$1
set -e
# 只要有 process 不成功，shell将会自动结束进程，不会继续执行

# 首先，训练1/8分辨率的位姿配准结果并渲染
echo "Outer dataset path: $1"
echo "[INFO] Training 1/8 resolution models for pose optimization."
./train_all_no_skew.sh $1
echo "[INFO] Training 1/8 resolution models completed."
# 训练完渲染，结果暂存在 renders 文件夹中

# no_skew dataparser_transforms.json 需要处理
# 生成 output_no_skew.json
# 最后一个参数为 process_daya, 设为 false
echo "[INFO] Generating output.json for no_skew"
./output_json_gen.sh $1 _no_skew 0.125 0
echo "[INFO] Generating output_no_skew.json generated."

echo "[INFO] Rendering 1/8 resolution models..."
./render_all_no_skew.sh $1
echo "[INFO] Rendering 1/8 resolution models completed."

# 训练实际的 model，此model训练完后其实已经有结果的模型以供 ns-viewer 进行查看了
echo "[INFO] Traning model..."
./train_all.sh $1
echo "[INFO] Traning model completed."

# TODO: 生成位姿配准数据集，此时dayatemple需要处理，故设为 1
# 1. 生成 output_xxxx.json
echo "[INFO] Generating output.json for all models"
./output_json_gen.sh $1 _colmap 1.0 1 
echo "[INFO] Generating output.json completed."

# 2. 生成完整的测试位姿优化数据集，包括 transforms_xxx_opt.json

echo "[INFO] Camera test view optimization dataset started."
./create_cam_opt.sh $1
echo "[INFO] Camera test view optimization dataset completed."

# 3. 测试位姿优化的训练
echo "[INFO] Pose optimization started."
./pose_opt_all.sh $1
echo "[INFO] Pose optimization completed."

# 4. pose_replacer 得到 output_xxxx_opt.json
# 注意，此中需要进行处理: DayaTemple 与 PeonyGarden 有需要替换的部分

echo "[INFO] Pose placing started."
./opt_pose_replacer.sh $1
echo "[INFO] Pose placing completed."
echo "[INFO] Training completed."
