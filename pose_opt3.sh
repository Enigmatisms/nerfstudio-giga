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

opt_ids=(6 7)
for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}
    image_num=${image_nums[$idx]}
    CUDA_VISIBLE_DEVICES=3 ns-train depth-nerfacto \
        --data ../dataset/${folder}_opt/transforms${1}_opt.json \
        --load-dir ./outputs/$folder/depth-nerfacto/${folder}${1}${2}/nerfstudio_models/ \
        --timestamp ${folder}${1}${2} \
        --logging.local-writer.max-log-size 10 \
        --pipeline.model.log2-hashmap-size 21 \
        --pipeline.model.hidden-dim 128 \
        --pipeline.model.hidden-dim-color 128 \
        --pipeline.model.hidden-dim-transient 128 \
        --pipeline.model.num-levels 18 \
        --pipeline.model.max-res 3000 \
        --pipeline.model.num-proposal-samples-per-ray 512 256 \
        --pipeline.model.num-nerf-samples-per-ray 128 \
        --pipeline.model.background-color last_sample \
        --pipeline.model.original-image-num $image_num \
        --pipeline.datamanager.skip-eval True \
        --pipeline.datamanager.intrinsic-scale-factor 0.25 \
        --viewer.quit-on-train-completion True \
        --vis viewer+tensorboard \
        --pipeline.model.freeze-field True \
        --optimizers.fields.optimizer.lr 0 \
        --optimizers.proposal-networks.optimizer.lr 0 \
        --pipeline.datamanager.camera-optimizer.mode SE3 \
        --pipeline.datamanager.camera-optimizer.intrinsic-opt ${int_mode[$idx]} \
        --pipeline.datamanager.camera-optimizer.distortion-opt ${int_mode[$idx]} \
        --pipeline.datamanager.camera-optimizer.scheduler.lr-final 5e-5 \
        --pipeline.datamanager.camera-optimizer.scheduler.max-steps 36000 \
        --pipeline.datamanager.transform_path ./outputs/$folder/depth-nerfacto/${folder}${1}${2}/dataparser_transforms.json \
        --max-num-iterations 40000
done
