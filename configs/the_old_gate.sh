file_name="transforms${2}"
folder_name="old_gate${2}"
full_res_name="theOldGate${2}"

ns-train depth-nerfacto \
    --data ${1}/theOldGate/$file_name.json \
    --timestamp $folder_name \
    --logging.local-writer.max-log-size 10 \
    --pipeline.model.near-plane 0.1 \
    --pipeline.model.log2-hashmap-size 19 \
    --pipeline.model.hidden-dim 64 \
    --pipeline.model.distortion-loss-mult 0.002 \
    --pipeline.model.num-nerf-samples-per-ray 64 \
    --pipeline.model.num-levels 17 \
    --pipeline.model.orientation-loss-mult 0.0001 \
    --pipeline.model.proposal-update-every 5 \
    --pipeline.model.predict-normals False \
    --pipeline.model.background-color last_sample \
    --pipeline.model.use-entropy-loss True \
    --pipeline.model.entropy-threshold 0.01 \
    --pipeline.model.entropy-loss-mult 0.0005 \
    --pipeline.model.use-occ-regularization True \
    --pipeline.model.min-occ-threshold 0.1 \
    --pipeline.model.max-occ-threshold 0.4 \
    --pipeline.model.min-occ-loss_mult 0.0002 \
    --pipeline.model.max-occ-loss_mult 0.001 \
    --pipeline.model.occ-reg-iters 2000 \
    --pipeline.model.sigma-perturb-std 0.0 \
    --pipeline.model.sigma-perturb-iter 0 \
    --pipeline.model.min-depth-loss-mult 1e-3 \
    --pipeline.model.max-depth-loss-mult 1e-3 \
    --pipeline.model.depth-loss-iter 100000 \
    --pipeline.model.depth-sigma 0.01 \
    --pipeline.model.depth-loss-type DS_NERF \
    --pipeline.datamanager.skip-eval True \
    --pipeline.model.sample-unseen-views True \
    --pipeline.model.kl-divergence-mult 0.1 \
    --pipeline.model.test-occ-loss-mult 0.05 \
    --pipeline.model.test-near-plane 0.02 \
    --pipeline.model.test-far-plane 0.2 \
    --pipeline.datamanager.test-view-sample-iter -1 \
    --pipeline.datamanager.intrinsic-scale-factor 0.125 \
    --pipeline.datamanager.camera-optimizer.mode off \
    --pipeline.datamanager.unseen-sample-iter 10000 \
    --pipeline.datamanager.perturb-rot-sigma 5.0 \
    --pipeline.datamanager.unseen-ratio 1.0 \
    --pipeline.datamanager.sample_unseen_view True \
    --viewer.quit-on-train-completion True \
    --vis viewer+tensorboard \
    --max-num-iterations 40000

ns-train depth-nerfacto \
    --data ${1}/theOldGate/$file_name.json \
    --load-dir ./outputs/theOldGate/depth-nerfacto/$folder_name/nerfstudio_models/ \
    --timestamp $full_res_name \
    --logging.local-writer.max-log-size 10 \
    --pipeline.model.near-plane 0.1 \
    --pipeline.model.log2-hashmap-size 19 \
    --pipeline.model.hidden-dim 64 \
    --pipeline.model.distortion-loss-mult 1e-4 \
    --pipeline.model.num-nerf-samples-per-ray 64 \
    --pipeline.model.num-levels 17 \
    --pipeline.model.orientation-loss-mult 0.0001 \
    --pipeline.model.proposal-update-every 5 \
    --pipeline.model.predict-normals False \
    --pipeline.model.background-color last_sample \
    --pipeline.model.use-entropy-loss True \
    --pipeline.model.entropy-threshold 0.01 \
    --pipeline.model.entropy-loss-mult 0.0005 \
    --pipeline.model.use-occ-regularization False \
    --pipeline.model.min-occ-threshold 0.1 \
    --pipeline.model.max-occ-threshold 0.2 \
    --pipeline.model.min-occ-loss_mult 0.0001 \
    --pipeline.model.max-occ-loss_mult 0.0002 \
    --pipeline.model.occ-reg-iters 1000 \
    --pipeline.model.sigma-perturb-std 0.0 \
    --pipeline.model.sigma-perturb-iter 0 \
    --pipeline.model.min-depth-loss-mult 2e-3 \
    --pipeline.model.max-depth-loss-mult 2e-3 \
    --pipeline.model.depth-loss-iter 100000 \
    --pipeline.model.depth-sigma 0.01 \
    --pipeline.model.depth-loss-type DS_NERF \
    --pipeline.model.sample-unseen-views False \
    --pipeline.model.kl-divergence-mult 0.1 \
    --pipeline.datamanager.skip-eval True \
    --pipeline.datamanager.intrinsic-scale-factor 0.25 \
    --viewer.quit-on-train-completion True \
    --vis viewer+tensorboard \
    --pipeline.model.loss-coefficients.rgb-loss-coarse 0.5 \
    --optimizers.fields.optimizer.lr 5e-3 \
    --optimizers.proposal-networks.optimizer.lr 5e-3 \
    --max-num-iterations 80000