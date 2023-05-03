# Training procedures, for specific scene please go to the ./configs/
files=("science_square")
for file in ${files[@]}; do
    echo "Processing $file.sh"
    # Usage: nothing for transforms.json, _new for transforms_new.json, _no_skew, _colmap for ...
    CUDA_VISIBLE_DEVICES=1 ./config1/$file.sh ../dataset/ _colmap
done