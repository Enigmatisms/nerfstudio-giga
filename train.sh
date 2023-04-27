# Training procedures, for specific scene please go to the ./configs/
files=("haiyan_hall" "library" "memorial_hall")
for file in ${files[@]}; do
    echo "Processing $file.sh"
    CUDA_VISIBLE_DEVICES=0 ./configs/$file.sh ../dataset/
done