# CUDA_VISIBLE_DEVICES=1 ns-train depth-nerfacto \
#     --data ../dataset/daya/daya_igev/transforms.json \
#     --logging.local-writer.max-log-size 10 \
#     --pipeline.model.log2-hashmap-size 19 \
#     --pipeline.model.hidden-dim 64 \
#     --pipeline.model.distortion-loss-mult 0.002 \
#     --pipeline.model.num-nerf-samples-per-ray 48 \
#     --pipeline.model.orientation-loss-mult 0.0001 \
#     --pipeline.model.proposal-update-every 5 \
#     --pipeline.model.predict-normals False \
#     --pipeline.model.background-color last_sample \
#     --pipeline.model.use-entropy-loss False \
#     --pipeline.model.entropy-threshold 0.005 \
#     --pipeline.model.entropy-loss-mult 0.001 \
#     --pipeline.model.use-occ-regularization False \
#     --pipeline.model.min-occ-threshold 0.1 \
#     --pipeline.model.max-occ-threshold 0.2 \
#     --pipeline.model.min-occ-loss_mult 0.0001 \
#     --pipeline.model.max-occ-loss_mult 0.0005 \
#     --pipeline.model.occ-reg-iters 1000 \
#     --pipeline.model.sigma-perturb-std 0.0 \
#     --pipeline.model.sigma-perturb-iter 0 \
#     --pipeline.model.min-depth-loss-mult 1e-3 \
#     --pipeline.model.max-depth-loss-mult 1e-3 \
#     --pipeline.model.depth-loss-iter 100000 \
#     --pipeline.model.depth-sigma 0.01 \
#     --pipeline.model.depth-loss-type DS_NERF \
#     --pipeline.model.sample-unseen-views False \
#     --pipeline.model.kl-divergence-mult 0.1 \
#     --pipeline.datamanager.skip-eval True \
#     --pipeline.datamanager.intrinsic-scale-factor 0.125 \
#     --pipeline.datamanager.camera-optimizer.mode off \
#     --viewer.quit-on-train-completion True \
#     --max-num-iterations 30000

CUDA_VISIBLE_DEVICES=1 ns-train depth-nerfacto \
    --data ../dataset/daya/daya_igev/transforms.json \
    --load-dir ./outputs/daya_igev/depth-nerfacto/2023-04-23_125127/nerfstudio_models/ \
    --logging.local-writer.max-log-size 10 \
    --pipeline.model.log2-hashmap-size 19 \
    --pipeline.model.hidden-dim 64 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.num-nerf-samples-per-ray 48 \
    --pipeline.model.orientation-loss-mult 0.0001 \
    --pipeline.model.proposal-update-every 5 \
    --pipeline.model.predict-normals False \
    --pipeline.model.background-color last_sample \
    --pipeline.model.use-entropy-loss False \
    --pipeline.model.entropy-threshold 0.005 \
    --pipeline.model.entropy-loss-mult 0.001 \
    --pipeline.model.use-occ-regularization False \
    --pipeline.model.min-occ-threshold 0.1 \
    --pipeline.model.max-occ-threshold 0.2 \
    --pipeline.model.min-occ-loss_mult 0.0001 \
    --pipeline.model.max-occ-loss_mult 0.0005 \
    --pipeline.model.occ-reg-iters 1000 \
    --pipeline.model.sigma-perturb-std 0.0 \
    --pipeline.model.sigma-perturb-iter 0 \
    --pipeline.model.min-depth-loss-mult 1e-3 \
    --pipeline.model.max-depth-loss-mult 1e-3 \
    --pipeline.model.depth-loss-iter 100000 \
    --pipeline.model.depth-sigma 0.01 \
    --pipeline.model.depth-loss-type DS_NERF \
    --pipeline.model.sample-unseen-views False \
    --pipeline.model.kl-divergence-mult 0.1 \
    --pipeline.datamanager.skip-eval True \
    --pipeline.datamanager.intrinsic-scale-factor 0.25 \
    --pipeline.datamanager.camera-optimizer.mode off \
    --viewer.quit-on-train-completion True \
    --max-num-iterations 60000