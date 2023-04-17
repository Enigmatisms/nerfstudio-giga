ns-process-data images \
    --data ../dataset/images_and_cams/HaiyanHall/images/ \
    --output-dir ../dataset/images_and_cams/HaiyanHall/image_colmap/ \
    --sfm-tool colmap --use-sfm-depth

# ns-process-data images \
#     --data ../dataset/images_and_cams/PeonyGarden/image_scaled/ \
#     --output-dir ../dataset/images_and_cams/PeonyGarden/image_hloc/ \
#     --sfm-tool hloc --refine-pixsfm --use-sfm-depth