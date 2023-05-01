folders=("HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare")

if [ ""$1 = "" ]; then
    echo "You have specified no <SUFFIX>, which can be ['', '_new', '_no_skew']"
    echo "Normally, '_new' will be your choice."
    echo "For example: <./render.sh _new> will mean you choose to render with original poses and COLMAP intrinsics."
fi

for folder in ${folders[@]}; do
    recently_created=`ls -td -- ./outputs/$folder/depth-nerfacto/*/ | head -n 1`
    # second_created=`ls -td -- ./outputs/$folder/depth-nerfacto/*/ | head -n 2 | tail -n 1`
    folder_name=${recently_created#*/}
    echo "Rendering $folder_name"
    CUDA_VISIBLE_DEVICES=0 ns-render \
        --load-config ${folder_name}config.yml \
        --traj filename --camera-path-filename ../dataset/images_and_cams/full/pose_align/$folder/output$1.json \
        --output-path renders/$folder/ --output_format images
done

# CUDA_VISIBLE_DEVICES=0 python ./train.py
