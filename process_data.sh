# ns-process-data images \
#     --data ../dataset/images_and_cams/Library/image_scaled/ \
#     --output-dir ../dataset/images_and_cams/Library/image_colmap/ \
#     --sfm-tool colmap --use-sfm-depth

# ns-process-data images \
#     --data ../dataset/images_and_cams/MemorialHall/image_scaled/ \
#     --output-dir ../dataset/images_and_cams/MemorialHall/memorial_hloc/ \
#     --sfm-tool hloc --refine-pixsfm --use-sfm-depth

# ns-process-data images \
#     --data ../dataset/images_and_cams/Museum/image_scaled/ \
#     --output-dir ../dataset/images_and_cams/Museum/museum_hloc/ \
#     --sfm-tool hloc --refine-pixsfm --use-sfm-depth

ns-process-data images \
    --data ../dataset/images_and_cams/theOldGate/image_scaled/ \
    --output-dir ../dataset/images_and_cams/theOldGate/gate_hloc/ \
    --sfm-tool hloc --refine-pixsfm