# Training procedures, for specific scene please go to the ./configs/

CUDA_VISIBLE_DEVICES=3 ./configs/haiyan_hall.sh ../dataset/ _colmap
CUDA_VISIBLE_DEVICES=3 ./configs/daya_temple.sh ../dataset/