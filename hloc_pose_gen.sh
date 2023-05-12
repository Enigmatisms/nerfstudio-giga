input_folder=$1
output_folder=$2
folder1=$3
folder2=$4

if [ ! -d $input_folder ]; then
    echo "[HLOC POSE - $5] Input folder '${input_folder}' does not exist, please check."
fi

CUDA_VISIBLE_DEVICES=$5 ns-process-data images \
    --data $input_folder$folder1/ \
    --output-dir $output_folder$folder1/ \
    --sfm-tool hloc --refine-pixsfm

CUDA_VISIBLE_DEVICES=$5 ns-process-data images \
    --data $input_folder$folder2/ \
    --output-dir $output_folder$folder2/ \
    --sfm-tool hloc --refine-pixsfm


echo "[HLOC POSE - $5] HLOC pose generation process $5 completed."