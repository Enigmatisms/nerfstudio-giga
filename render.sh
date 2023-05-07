folders=("theOldGate") #"Library") #"MemorialHall" "HaiyanHall")  # 160 for newest config rendering in cuda 1

if [ ""$1 = "" ]; then
    echo "You have specified no <SUFFIX>, which can be ['', '_new', '_no_skew']"
    echo "Or something with `_opt` tag"
    echo "Normally, '_new' will be your choice."
    echo "For example: <./render.sh _new> will mean you choose to render with original poses and COLMAP intrinsics."
fi

length=${#folders[@]}
for ((i=0;i<$length;i++)); do
    folder=${folders[$i]}
    model_folder=${folders[$i]}${1}_high

    folder_name=./outputs/$folder/depth-nerfacto/$model_folder/
    echo "Rendering $folder_name"
    CUDA_VISIBLE_DEVICES=0 ns-render \
        --load-config ${folder_name}config.yml \
        --traj filename --camera-path-filename ../dataset/$folder/output${1}_opt.json \
        --output-path renders/$folder/ --output_format images \
        --eval_num_rays_per_chunk 4096
done

# CUDA_VISIBLE_DEVICES=0 python ./train.py
