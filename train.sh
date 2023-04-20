ns-train depth-nerfacto \
    --data ../dataset/images_and_cams/HaiyanHall/image_hloc/transforms.json \
    --pipeline.model.log2-hashmap-size 19 \
    --pipeline.model.hidden-dim 64 \
    --pipeline.model.distortion-loss-mult 0.00000001 \
    --pipeline.model.num-nerf-samples-per-ray 56 \
    --pipeline.model.orientation-loss-mult 0.0001 \
    --pipeline.model.proposal-update-every 4 \
    --pipeline.model.predict-normals False \
    --pipeline.model.collider-params near_plane 0.1 far_plane 10.0 \
    --pipeline.model.background-color last_sample \
    --pipeline.model.entropy-threshold 0.005 \
    --pipeline.model.entropy-loss-mult 0.01 \
    --pipeline.model.use-entropy-loss True \
    --pipeline.model.use-occ-regularization True \
    --pipeline.model.min-occ-threshold 0.1 \
    --pipeline.model.max-occ-threshold 0.4 \
    --pipeline.model.min-occ-loss_mult 0.0002 \
    --pipeline.model.max-occ-loss_mult 0.001 \
    --pipeline.model.occ-reg-iters 40000 \
    --pipeline.model.sigma-perturb-std 0.1 \
    --pipeline.model.sigma-perturb-iter 50000 \
    --pipeline.model.depth-loss-mult 0.002 \
    --pipeline.model.depth-sigma 0.04 \
    --max-num-iterations 300000
    # --load-dir ./outputs/DayaTemple/nerfacto/2023-04-18_094759/nerfstudio_models
    # --logging.local-writer.stats-to-track 