import cv2
import time
import argparse
import os
import torch
import json
import posenet
import shutil
import sklearn.metrics
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
parser.add_argument('--output_keypoint_json_file', type=str, default='./output_keypoint.json')
parser.add_argument('--output_keyframe_json_file', type=str, default='./output_keyframe.json')
args = parser.parse_args()

json_result_dict = {}
cosine_threshold = 0.7
keyframe_result = []

def extract_keyframe(keypoint_result_dict):
    cos_diff = []
    max_frame = len(keypoint_result_dict)
    frame_vect_list = []
    last_frame = 0
    for frame_i in range(1, max_frame):
        if len(keypoint_result_dict[frame_i]) == 0:
            continue
        result = keypoint_result_dict[frame_i]
        result_dict = {}
        current_frame_vect = np.zeros((0,2))
        
        good_point = 0
        for point in result:
            if point['score'] > 0.3:
                good_point+=1
            result_dict[point['part']] = [point['position']['x'], point['position']['y']]

        for point_pair in posenet.POSE_CHAIN:
            current_frame_vect = np.vstack((current_frame_vect, np.array(result_dict[point_pair[0]]) - np.array(result_dict[point_pair[1]])))

        if len(frame_vect_list) != 0:
            cosine_dist = [sklearn.metrics.pairwise.cosine_similarity([current_frame_vect[i]], [frame_vect_list[-1][i]]) for i in range(len(current_frame_vect))]
            cos_diff.append(cosine_dist)
            min_value = min(cosine_dist)
            min_index = {p_i for p_i in range(len(cosine_dist)) if cosine_dist[p_i] < cosine_threshold}
            # major movement
            if min_value < cosine_threshold and good_point > 14:
                # prove calculate
                if frame_i -3 > 0:
                    cosine_dist2 = [sklearn.metrics.pairwise.cosine_similarity([current_frame_vect[i]], [frame_vect_list[-3][i]]) for i in range(len(current_frame_vect))]
                    min_index2 = {p_i for p_i in range(len(cosine_dist2)) if cosine_dist2[p_i] < cosine_threshold}
                attension_set = min_index.union(min_index2)
                if frame_i -last_frame > 5:
                    keyframe_result.append({frame_i: [posenet.POSE_CHAIN[p_i] for p_i in attension_set]})
                last_frame = frame_i
            # pause no more than 60 frames, 3 sec
            elif frame_i - last_frame > 60:
                last_frame = frame_i
                keyframe_result.append({frame_i: []})

        frame_vect_list.append(current_frame_vect)

    with open(args.output_keyframe_json_file, 'w') as outfile:
        json.dump(keyframe_result, outfile)

def extract_keypoints():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    if args.output_dir:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

    filenames = [
        f.path for f in os.scandir(args.image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]

    start = time.time()
    filenames.sort()
    json_result_dict = {}
    for f in filenames:
        input_image, draw_image, output_scale = posenet.read_imgfile(
            f, scale_factor=args.scale_factor, output_stride=output_stride)

        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        keypoint_coords *= output_scale

        if args.output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.25)

            cv2.imwrite(os.path.join(args.output_dir, os.path.relpath(f, args.image_dir)), draw_image)

        if not args.notxt:
            print()
            print("Results for image: %s" % f)

            frame_num = int(os.path.basename(f).split('.')[0])

            result = []
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                    result.append({'score': s,
                        "part":posenet.PART_NAMES[ki],
                        "position":{'x':c[1], 'y':c[0]}})
            json_result_dict[frame_num] = result

    with open(args.output_keypoint_json_file, 'w') as outfile:
        json.dump(json_result_dict, outfile)

    print('Average FPS:', len(filenames) / (time.time() - start))

    return json_result_dict


def main():
    result_dict = extract_keypoints()
    extract_keyframe(result_dict)
    

if __name__ == "__main__":
    main()
