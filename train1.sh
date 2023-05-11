# Training procedures, for specific scene please go to the ./configs/
files=("museum")
for file in ${files[@]}; do
    echo "Processing $file.sh"
    CUDA_VISIBLE_DEVICES=1 ./config1/$file.sh ../dataset/ _no_skew
doned