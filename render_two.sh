scene1=$1
scene2=$2
data_set_path=$3
mode=$4
suffix=$5

input_path1=./outputs/$scene1/depth-nerfacto/${scene1}${mode}/
input_path2=./outputs/$scene2/depth-nerfacto/${scene2}${mode}/

ns-render \
        --load-config ${input_path1}config.yml \
        --traj filename --camera-path-filename ${data_set_path}${scene1}/output${mode}${suffix}.json \
        --output-path renders/$scene1/ --output_format images \
        --eval_num_rays_per_chunk 4096

ns-render \
        --load-config ${input_path2}config.yml \
        --traj filename --camera-path-filename ${data_set_path}${scene2}/output${mode}${suffix}.json \
        --output-path renders/$scene2/ --output_format images \
        --eval_num_rays_per_chunk 4096