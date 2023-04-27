# Training procedures, for specific scene please go to the ./configs/

for file in `ls ./configs/`; do
    echo "Processing $file"
    ./configs/$file ../dataset/images_and_cams/full/
done