scene1=$1
data_set_path=$2
mode=$3
near=$4
sample_num=$5
origin_img_num=$6
suffix=$7

input_path=./outputs/$scene1/depth-nerfacto/${scene1}${mode}/

ns-render \
        --load-config ${input_path}config.yml \
        --traj filename --camera-path-filename ${data_set_path}${scene1}/output${mode}${suffix}.json \
        --output-path renders/$scene1/ --output_format images \
        --eval_num_rays_per_chunk 4096 --render-sample-num $sample_num --original-img-num $origin_img_num --near-plane $near
