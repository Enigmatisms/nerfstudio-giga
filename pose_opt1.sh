if [ ""$1 = "" ]; then
    echo "You have specified no <SUFFIX>, which can be ['', '_new', '_no_skew']"
    echo "Or something with `_opt` tag"
    echo "Normally, '_new' will be your choice."
    echo "For example: <./render.sh _new> will mean you choose to render with original poses and COLMAP intrinsics."
fi

if [ ""$2 = "" ]; then
    echo "You might want to fill in the second argument."
    echo "For exmaple: ./pose_opt.sh _colmap _1."
fi

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
image_nums=(83 25 53 17 54 42 32 47)
int_mode=("off" "fixed" "fixed" "fixed" "fixed" "off" "fixed" "fixed")


opt_ids=(4)
for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}
    image_num=${image_nums[$idx]}
    CUDA_VISIBLE_DEVICES=0,1 ns-train depth-nerfacto \
        --data ../dataset/${folder}_opt/transforms${1}_opt.json \
        --load-dir ./outputs/$folder/depth-nerfacto/${folder}${1}${2}/nerfstudio_models/ \
        --timestamp ${folder}${1}${2} \
        --logging.local-writer.max-log-size 10 \
        --pipeline.model.num-nerf-samples-per-ray 96 \
        --pipeline.model.log2-hashmap-size 19 \
        --pipeline.model.hidden-dim 64 \
        --pipeline.model.near-plane 0.0 \
        --pipeline.model.far-plane 80 \
        --pipeline.model.num-levels 17 \
        --pipeline.model.background-color last_sample \
        --pipeline.model.original-image-num $image_num \
        --pipeline.datamanager.skip-eval True \
        --pipeline.datamanager.intrinsic-scale-factor 0.125 \
        --viewer.quit-on-train-completion True \
        --vis viewer+tensorboard \
        --pipeline.model.freeze-field True \
        --optimizers.fields.optimizer.lr 0 \
        --optimizers.proposal-networks.optimizer.lr 0 \
        --pipeline.datamanager.camera-optimizer.mode SE3 \
        --pipeline.datamanager.camera-optimizer.intrinsic-opt ${int_mode[$idx]} \
        --pipeline.datamanager.camera-optimizer.distortion-opt ${int_mode[$idx]} \
        --pipeline.datamanager.camera-optimizer.optimizer.lr 1e-4 \
        --pipeline.datamanager.camera-optimizer.scheduler.lr-final 5e-5 \
        --pipeline.datamanager.camera-optimizer.scheduler.max-steps 9000 \
        --pipeline.datamanager.transform_path ./outputs/$folder/depth-nerfacto/${folder}${1}${2}/dataparser_transforms.json \
        --max-num-iterations 12000
done

cd pose_viz/
python3 ./replace_pose.py -s Museum_opt -n Museum_no_skew -m "no_skew" -o ../../dataset/ --scale 8.0
cd ..