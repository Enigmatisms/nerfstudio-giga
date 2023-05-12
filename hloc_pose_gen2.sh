input_folder=$1
output_folder=$2

if [ ! -d $input_folder ]; then
    echo "[HLOC POSE - 2] Input folder '${input_folder}' does not exist, please check."
fi

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
opt_ids=(4 5 6 7)
for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}
    CUDA_VISIBLE_DEVICES=1 ns-process-data images \
        --data $input_folder$folder/ \
        --output-dir $output_folder$folder/ \
        --sfm-tool hloc --refine-pixsfm
done

echo "[HLOC POSE - 2] HLOC pose generation process 2 completed."