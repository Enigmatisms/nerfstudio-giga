
data_set_path=$1

echo "[POSE OPT] Test view pose optimizatin started."

folders=("DayaTemple" "HaiyanHall" "Library" "MemorialHall" "Museum" "PeonyGarden" "ScienceSquare" "theOldGate")
image_nums=(83 25 53 17 54 42 32 47)
int_mode=("off" "fixed" "fixed" "fixed" "fixed" "off" "fixed" "fixed")
opt_modes=("" "_colmap" "_colmap" "_colmap" "_colmap" "_colmap" "_colmap" "_colmap")

opt_ids=(4 5 6 7)
for idx in ${opt_ids[@]}; do
    folder=${folders[$idx]}
    opt_mode=${opt_modes[$idx]}
    image_num=${image_nums[$idx]}
    CUDA_VISIBLE_DEVICES=1 ns-train depth-nerfacto \
        --data ${data_set_path}${folder}_opt/transforms${opt_mode}_opt.json \
        --load-dir ./outputs/$folder/depth-nerfacto/${folder}${opt_mode}/nerfstudio_models/ \
        --timestamp ${folder} \
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
        --pipeline.datamanager.camera-optimizer.scheduler.lr-final 5e-5 \
        --pipeline.datamanager.camera-optimizer.scheduler.max-steps 9000 \
        --pipeline.datamanager.transform_path ./outputs/$folder/depth-nerfacto/${folder}${opt_mode}/dataparser_transforms.json \
        --max-num-iterations 12000
done

echo "[POSE OPT] Test view pose optimizatin completed."
