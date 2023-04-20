ns-train depth-nerfacto \
    --data ../dataset/PeonyGarden/transforms.json \
    --pipeline.model.log2-hashmap-size 19 \
    --pipeline.model.hidden-dim 64 \
    --pipeline.model.distortion-loss-mult 0.00000001 \
    --pipeline.model.num-nerf-samples-per-ray 64 \
    --pipeline.model.orientation-loss-mult 0.0001 \
    --pipeline.model.proposal-update-every 4 \
    --pipeline.model.predict-normals False \
    --pipeline.model.collider-params near_plane 0.1 far_plane 10.0 \
    --pipeline.model.background-color last_sample \
    --pipeline.model.entropy-threshold 0.005 \
    --pipeline.model.entropy-loss-mult 0.005 \
    --pipeline.model.use-entropy-loss True \
    --pipeline.model.use-occ-regularization True \
    --pipeline.model.min-occ-threshold 0.05 \
    --pipeline.model.max-occ-threshold 0.2 \
    --pipeline.model.min-occ-loss_mult 0.0001 \
    --pipeline.model.max-occ-loss_mult 0.0005 \
    --pipeline.model.occ-reg-iters 50000 \
    --pipeline.model.sigma-perturb-std 0.125 \
    --pipeline.model.sigma-perturb-iter 80000 \
    --pipeline.model.depth-loss-mult 0.003 \
    --pipeline.model.depth-sigma 0.03 \
    --max-num-iterations 300000
    # --load-dir ./outputs/DayaTemple/nerfacto/2023-04-18_094759/nerfstudio_models
    # --logging.local-writer.stats-to-track 

echo "One model is completed."

ns-train depth-nerfacto \
    --data ../dataset/ScienceSquare/transforms.json \
    --pipeline.model.log2-hashmap-size 19 \
    --pipeline.model.hidden-dim 64 \
    --pipeline.model.distortion-loss-mult 0.00000001 \
    --pipeline.model.num-nerf-samples-per-ray 64 \
    --pipeline.model.orientation-loss-mult 0.0001 \
    --pipeline.model.proposal-update-every 4 \
    --pipeline.model.predict-normals False \
    --pipeline.model.collider-params near_plane 0.1 far_plane 10.0 \
    --pipeline.model.background-color last_sample \
    --pipeline.model.entropy-threshold 0.005 \
    --pipeline.model.entropy-loss-mult 0.005 \
    --pipeline.model.use-entropy-loss True \
    --pipeline.model.use-occ-regularization True \
    --pipeline.model.min-occ-threshold 0.1 \
    --pipeline.model.max-occ-threshold 0.3 \
    --pipeline.model.min-occ-loss_mult 0.0002 \
    --pipeline.model.max-occ-loss_mult 0.0007 \
    --pipeline.model.occ-reg-iters 50000 \
    --pipeline.model.sigma-perturb-std 0.125 \
    --pipeline.model.sigma-perturb-iter 80000 \
    --pipeline.model.depth-loss-mult 0.002 \
    --pipeline.model.depth-sigma 0.03 \
    --max-num-iterations 300000