scene1=$1
data_set_path=$2
mode=$3
suffix=$4

input_path=./outputs/$scene1/depth-nerfacto/${scene1}${mode}/

ns-render \
        --load-config ${input_path}config.yml \
        --traj filename --camera-path-filename ${data_set_path}${scene1}/output${mode}${suffix}.json \
        --output-path renders/$scene1/ --output_format images \
        --eval_num_rays_per_chunk 4096
