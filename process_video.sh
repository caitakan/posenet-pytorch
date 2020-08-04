echo "label quality also depend on the complexity of the video"
echo "process the video $1 to keypoint file $2 keyframe file $3"

rm -r temp_output_image
mkdir temp_output_image
ffmpeg -i $1 -vf fps=20 temp_output_image/%04d.png
python write_keypoint_json.py --model 101 --image_dir temp_output_image/ --output_keypoint_json_file $2 --output_keyframe_json_file $3
