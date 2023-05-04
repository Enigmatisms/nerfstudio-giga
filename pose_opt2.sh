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
levels=(16 17 17 17 17 17 17 17)

opt_ids=(4 5)
for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}
    image_num=${image_nums[$idx]}
    level=${levels[$idx]}
    CUDA_VISIBLE_DEVICES=2 ns-train depth-nerfacto \
        --data ../dataset/${folder}_opt/transforms${1}_opt.json \
        --load-dir ./outputs/$folder/depth-nerfacto/${folder}${1}${2}/nerfstudio_models/ \
        --timestamp ${folder}${1}${2} \
        --logging.local-writer.max-log-size 10 \
        --pipeline.model.num-nerf-samples-per-ray 64 \
        --pipeline.model.log2-hashmap-size 19 \
        --pipeline.model.hidden-dim 64 \
        --pipeline.model.near-plane 0.1 \
        --pipeline.model.num-levels $level \
        --pipeline.model.background-color last_sample \
        --pipeline.model.original-image-num $image_num \
        --pipeline.datamanager.skip-eval True \
        --pipeline.datamanager.intrinsic-scale-factor 0.125 \
        --pipeline.datamanager.camera-optimizer.mode off \
        --viewer.quit-on-train-completion True \
        --vis viewer+tensorboard \
        --pipeline.model.freeze-field True \
        --optimizers.fields.optimizer.lr 0 \
        --optimizers.proposal-networks.optimizer.lr 0 \
        --pipeline.datamanager.camera-optimizer.mode SE3 \
        --pipeline.datamanager.camera-optimizer.scheduler.lr-final 5e-5 \
        --pipeline.datamanager.camera-optimizer.scheduler.max-steps 8000 \
        --pipeline.datamanager.transform_path ./outputs/$folder/depth-nerfacto/${folder}${1}${2}/dataparser_transforms.json \
        --max-num-iterations 10000
done
