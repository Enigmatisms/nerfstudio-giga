# Training procedures, for specific scene please go to the ./configs/
files=("memorial_hall")
for file in ${files[@]}; do
    echo "Processing $file.sh"
    CUDA_VISIBLE_DEVICES=1 ./configs/$file.sh ../dataset/ _no_skew
done