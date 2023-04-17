ns-train nerfacto \
    --data ../dataset/images_and_cams/DayaTemple/train.json \
    --pipeline.model.log2-hashmap-size 19 \
    --pipeline.model.hidden-dim 64 \
    --pipeline.model.distortion-loss-mult 0.02 \
    --pipeline.model.num-nerf-samples-per-ray 48 \
    --pipeline.model.orientation-loss-mult 0.0001 \
    --pipeline.model.proposal-update-every 4 \
    --pipeline.model.predict-normals True \
    --pipeline.model.collider-params near_plane 0.1 far_plane 10.0 \
    --pipeline.model.background-color random \
    --pipeline.model.entropy-threshold 0.01 \
    --pipeline.model.entropy-loss-mult 0.002 \
    --pipeline.model.use-entropy-loss True
    # --logging.local-writer.stats-to-track 