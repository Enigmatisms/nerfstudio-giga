# data_set_path 需要输入 output.json 所在的位置

data_set_path=$1

echo "[POSE OPT GEN] Generating pose optimization dataset."

folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
# 输出将会是 transforms_colmap_opt.json
for folder in ${folders[@]}; do
    python3 ./pose_viz/camera_opt.py -i $folder -m colmap \
        --input_pose_path $data_set_path --output_path $data_set_path -f
done

# 当前在 code/nerfstudio/下, 将 daya_ngp_samples 中的图像复制到 Daya_opt 中用作位姿优化
# 此处并不违反规则，详情请见我们的README 1.5
if [ ! -d ./renders/DayaTemple/ ]; then
    mkdir -p ./renders/DayaTemple/
fi
cp ../../data/user_data/public_data/daya_ngp_samples/* ./renders/DayaTemple/

# 输出将会是 transforms_opt.json
python3 ./pose_viz/camera_opt.py -i DayaTemple -m none \
        --input_pose_path $data_set_path --output_path $data_set_path -f

echo "[POSE OPT GEN] Generating pose optimization dataset completed."