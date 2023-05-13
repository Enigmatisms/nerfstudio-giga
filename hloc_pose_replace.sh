input_folder=$1
output_folder=$2

if [ ! -d $input_folder ]; then
    echo "[HLOC REPLACE] Input folder '${input_folder}' does not exist, please check."
fi

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
opt_ids=(1 2 3 4 5 6)
for idx in ${opt_ids[@]}; do
    scene=${folders[$idx]}
    input_path=$input_folder$scene/
    output_path=$output_folder$scene/
    python3 ./pose_viz/hloc_poser.py -i $output_path --merge -o "transforms_colmap.json" --outpath $output_path  --scale 2.0 --overwrite               # 保留原本的 transforms.json
    python3 ./pose_viz/hloc_poser.py -i $output_path --merge --no_skew -o "transforms_no_skew.json" --outpath $output_path   # 生成 纯原始位姿 + 纯原始内参 json
    # 复制相关 json 到我们构建数据集的位置
done

# theOldGate 需要将其 applied transform 去掉 （theOldGate 位姿有一定特殊性）
python3 ./pose_viz/hloc_poser.py -i ${output_folder}theOldGate --transform --merge -o "transforms_colmap.json" --outpath ${output_folder}theOldGate/ --scale 2.0 --overwrite
python3 ./pose_viz/hloc_poser.py -i ${output_folder}theOldGate --transform --merge --no_skew -o "transforms_no_skew.json" --outpath ${output_folder}theOldGate/

# DayaTemple 没有 train test 合并，并且不需要替换位姿 (但是必须 overwrite 此前的内参，因为缩小了1/2)

python3 ./pose_viz/hloc_poser.py -i ${output_folder}DayaTemple -o "transforms.json" --outpath ${output_folder}DayaTemple/ --scale 2.0 --overwrite --skip

echo "[HLOC REPLACE] HLOC poses are replaced for 7 scenes, DayaTemple remains un-changed."