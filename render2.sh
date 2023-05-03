folders=("Museum") #"DayaTemple") # "PeonyGarden" "theOldGate" )  # 160 for newest2 config rendering in cuda 3
model_name=("2023-05-03_043550") #"2023-05-03_042849") # "2023-05-03_062922" "2023-05-03_103507" )


if [ ""$1 = "" ]; then
    echo "You have specified no <SUFFIX>, which can be ['', '_new', '_no_skew']"
    echo "Or something with `_opt` tag"
    echo "Normally, '_new' will be your choice."
    echo "For example: <./render.sh _new> will mean you choose to render with original poses and COLMAP intrinsics."
fi

length=${#folders[@]}
for ((i=0;i<$length;i++)); do
    # recently_created=`ls -td -- ./outputs/$folder/depth-nerfacto/*/ | head -n 1`
    # second_created=`ls -td -- ./outputs/$folder/depth-nerfacto/*/ | head -n 2 | tail -n 1`
    # folder_name=${second_created#*/}
    folder=${folders[$i]}
    model_folder=${model_name[$i]}

    folder_name=./outputs/$folder/depth-nerfacto/$model_folder/
    echo "Rendering $folder_name"
    CUDA_VISIBLE_DEVICES=3 ns-render \
        --load-config ${folder_name}config.yml \
        --traj filename --camera-path-filename ../dataset/$folder/output$1.json \
        --output-path renders/$folder/ --output_format images \
        --eval_num_rays_per_chunk 4096
done

# CUDA_VISIBLE_DEVICES=0 python ./train.py
