folders=("theOldGate")

for folder in ${folders[@]}; do
    python3 ./camera_opt.py -i $folder -m none -f --no_image
    # python3 ./camera_opt.py -i $folder -m colmap -f
done