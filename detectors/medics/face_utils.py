import numpy as np
from PIL import Image
import cv2
import math
import skimage.measure
from constants import *


def load_video(filename, every_n_frames=None, specific_frames=None, to_rgb=True, rescale=None, inc_pil=False, max_frames=None):
    """Loads a video.
    Called by:

    1) The finding faces algorithm where it pulls a frame every FACE_FRAMES frames up to MAX_FRAMES_TO_LOAD at a scale of FACEDETECTION_DOWNSAMPLE, and then half that if there's a CUDA memory error.

    2) The inference loop where it pulls EVERY frame up to a certain amount which it the last needed frame for each face for that video"""

    assert every_n_frames or specific_frames, "Must supply either every n_frames or specific_frames"
    assert bool(every_n_frames) != bool(
        specific_frames), "Supply either 'every_n_frames' or 'specific_frames', not both"

    cap = cv2.VideoCapture(filename)
    n_frames_in = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width_in = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height_in = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if rescale:
        rescale = rescale * 1920./np.max((width_in, height_in))

    width_out = int(width_in*rescale) if rescale else width_in
    height_out = int(height_in*rescale) if rescale else height_in

    if max_frames:
        n_frames_in = min(n_frames_in, max_frames)

    if every_n_frames:
        specific_frames = list(range(0, n_frames_in, every_n_frames))

    n_frames_out = len(specific_frames)

    out_pil = []

    out_video = np.empty(
        (n_frames_out, height_out, width_out, 3), np.dtype('uint8'))

    i_frame_in = 0
    i_frame_out = 0
    ret = True

    while (i_frame_in < n_frames_in and ret):

        try:
            try:

                if every_n_frames == 1:
                    ret, frame_in = cap.read()  # Faster if reading all frames
                else:
                    ret = cap.grab()

                    if i_frame_in not in specific_frames:
                        i_frame_in += 1
                        continue

                    ret, frame_in = cap.retrieve()

#                 print(f"Reading frame {i_frame_in}")

                if rescale:
                    frame_in = cv2.resize(frame_in, (width_out, height_out))
                if to_rgb:
                    frame_in = cv2.cvtColor(frame_in, cv2.COLOR_BGR2RGB)

            except Exception as e:
                print(
                    f"Error for frame {i_frame_in} for video {filename}: {e}; using 0s")
                frame_in = np.zeros((height_out, width_out, 3))

            out_video[i_frame_out] = frame_in
            i_frame_out += 1

            if inc_pil:
                try:  # https://www.kaggle.com/zaharch/public-test-errors
                    pil_img = Image.fromarray(frame_in)
                except Exception as e:
                    print(
                        f"Using a blank frame for video {filename} frame {i_frame_in} as error {e}")
                    pil_img = Image.fromarray(
                        np.zeros((224, 224, 3), dtype=np.uint8))  # Use a blank frame
                out_pil.append(pil_img)

            i_frame_in += 1

        except Exception as e:
            print(f"Error for file {filename}: {e}")

    cap.release()

    if inc_pil:
        return out_video, out_pil, rescale
    else:
        return out_video, rescale


def get_roi_for_each_face(faces_by_frame, probs, video_shape, temporal_upsample, upsample=1):
    # Create boolean face array
    frames_video, rows_video, cols_video, channels_video = video_shape
    frames_video = math.ceil(frames_video)
    boolean_face_3d = np.zeros(
        (frames_video, rows_video, cols_video), dtype=np.bool)  # Remove colour channel
    proba_face_3d = np.zeros(
        (frames_video, rows_video, cols_video)).astype('float32')
    for i_frame, faces in enumerate(faces_by_frame):
        if faces is not None:  # May not be a face in the frame
            for i_face, face in enumerate(faces):
                left, top, right, bottom = face
                boolean_face_3d[i_frame, int(top):int(
                    bottom), int(left):int(right)] = True
                proba_face_3d[i_frame, int(top):int(bottom), int(
                    left):int(right)] = probs[i_frame][i_face]

    # Replace blank frames if face(s) in neighbouring frames with overlap
    for i_frame, frame in enumerate(boolean_face_3d):
        if i_frame == 0 or i_frame == frames_video-1:  # Can't do this for 1st or last frame
            continue
        if True not in frame:
            if TWO_FRAME_OVERLAP:
                if i_frame > 1:
                    pre_overlap = boolean_face_3d[i_frame -
                                                  1] | boolean_face_3d[i_frame-2]
                else:
                    pre_overlap = boolean_face_3d[i_frame-1]
                if i_frame < frames_video-2:
                    post_overlap = boolean_face_3d[i_frame +
                                                   1] | boolean_face_3d[i_frame+2]
                else:
                    post_overlap = boolean_face_3d[i_frame+1]
                neighbour_overlap = pre_overlap & post_overlap
            else:
                neighbour_overlap = boolean_face_3d[i_frame -
                                                    1] & boolean_face_3d[i_frame+1]
            boolean_face_3d[i_frame] = neighbour_overlap

    # Find faces through time
    id_face_3d, n_faces = skimage.measure.label(
        boolean_face_3d, return_num=True)
    region_labels, counts = np.unique(id_face_3d, return_counts=True)
    # Get rid of background=0
    region_labels, counts = region_labels[1:], counts[1:]
    ###################
    # DESCENDING SIZE #
    ###################
    descending_size = np.argsort(counts)[::-1]
    labels_by_size = region_labels[descending_size]
    ####################
    # DESCENDING PROBS #
    ####################
    probs = [np.mean(proba_face_3d[id_face_3d == i_face])
             for i_face in region_labels]
    descending_probs = np.argsort(probs)[::-1]
    labels_by_probs = region_labels[descending_probs]
    # Iterate over faces in video
    rois = []
    face_maps = []
    for i_face in labels_by_probs:  # labels_by_size:
        # Find the first and last frame containing the face
        frames = np.where(np.any(id_face_3d == i_face, axis=(1, 2)) == True)
        starting_frame, ending_frame = frames[0].min(), frames[0].max()

        # Iterate over the frames with faces in and find the min/max cols/rows (bounding box)
        cols, rows = [], []
        for i_frame in range(starting_frame, ending_frame + 1):
            rs = np.where(
                np.any(id_face_3d[i_frame] == i_face, axis=1) == True)
            rows.append((rs[0].min(), rs[0].max()))
            cs = np.where(
                np.any(id_face_3d[i_frame] == i_face, axis=0) == True)
            cols.append((cs[0].min(), cs[0].max()))
        frame_from, frame_to = starting_frame * \
            temporal_upsample, ((ending_frame+1)*temporal_upsample)-1
        rows_from, rows_to = np.array(
            rows)[:, 0].min(), np.array(rows)[:, 1].max()
        cols_from, cols_to = np.array(
            cols)[:, 0].min(), np.array(cols)[:, 1].max()

        frame_to = min(frame_to, frame_from + MAX_FRAMES_FOR_FACE)

        if frame_to - frame_from >= MIN_FRAMES_FOR_FACE:
            tmp_face_map = id_face_3d.copy()
            tmp_face_map[tmp_face_map != i_face] = 0
            tmp_face_map[tmp_face_map == i_face] = 1
            face_maps.append(
                tmp_face_map[frame_from//temporal_upsample:frame_to//temporal_upsample+1])
            rois.append(((frame_from, frame_to),
                         (int(rows_from*upsample), int(rows_to*upsample)),
                         (int(cols_from*upsample), int(cols_to*upsample))))

    return np.array(rois), face_maps


def get_center(bbox):
    x1, y1, x2, y2 = bbox
    return (x1+x2)/2, (y1+y2)/2


def get_coords(faces_roi):
    coords = np.argwhere(faces_roi == 1)
    # print(coords)
    if coords.shape[0] == 0:
        return None
    y1, x1 = coords[0]
    y2, x2 = coords[-1]
    return x1, y1, x2, y2


def interpolate_center(c1, c2, length):
    x1, y1 = c1
    x2, y2 = c2
    xi, yi = np.linspace(x1, x2, length), np.linspace(y1, y2, length)
    return np.vstack([xi, yi]).transpose(1, 0)


def get_faces(faces_roi, upsample):
    all_faces = []
    rows = faces_roi[0].shape[1]
    cols = faces_roi[0].shape[2]
    for i in range(len(faces_roi)):
        faces = np.asarray([get_coords(faces_roi[i][j])
                            for j in range(len(faces_roi[i]))])
        if faces[0] is None:
            faces[0] = faces[1]
        if faces[-1] is None:
            faces[-1] = faces[-2]
        if None in faces:
            # print(faces)
            raise Exception('This should not have happened ...')
        all_faces.append(faces)

    extracted_faces = []
    for face in all_faces:
        # Get max dim size
        max_dim = np.concatenate(
            [face[:, 2]-face[:, 0], face[:, 3]-face[:, 1]])
        max_dim = np.percentile(max_dim, 90)
        # Enlarge by 1.2
        max_dim = int(max_dim * 1.2)
        # Get center coords
        centers = np.asarray([get_center(_) for _ in face])
        # Interpolate
        centers = np.vstack([interpolate_center(
            centers[i], centers[i+1], length=10) for i in range(len(centers)-1)]).astype('int')
        x1y1 = centers - max_dim // 2
        x2y2 = centers + max_dim // 2
        x1, y1 = x1y1[:, 0], x1y1[:, 1]
        x2, y2 = x2y2[:, 0], x2y2[:, 1]
        # If x1 or y1 is negative, turn it to 0
        # Then add to x2 y2 or y2
        x2[x1 < 0] -= x1[x1 < 0]
        y2[y1 < 0] -= y1[y1 < 0]
        x1[x1 < 0] = 0
        y1[y1 < 0] = 0
        # If x2 or y2 is too big, turn it to max image shape
        # Then subtract from y1
        y1[y2 > rows] += rows - y2[y2 > rows]
        x1[x2 > cols] += cols - x2[x2 > cols]
        y2[y2 > rows] = rows
        x2[x2 > cols] = cols
        vidface = np.asarray([[x1[_], y1[_], x2[_], y2[_]]
                              for _, c in enumerate(centers)])
        vidface = (vidface*upsample).astype('int')
        extracted_faces.append(vidface)

    return extracted_faces


def detect_face_with_mtcnn(mtcnn_model, pil_frames, facedetection_upsample, video_shape, face_frames):
    boxes, _probs = mtcnn_model.detect(pil_frames, landmarks=False)
    faces, faces_roi = get_roi_for_each_face(
        faces_by_frame=boxes, probs=_probs, video_shape=video_shape, temporal_upsample=face_frames, upsample=facedetection_upsample)
    coords = [] if len(faces_roi) == 0 else get_faces(
        faces_roi, upsample=facedetection_upsample)
    return faces, coords


def face_detection_wrapper(mtcnn_model, videopath, every_n_frames, facedetection_downsample, max_frames_to_load):
    video, pil_frames, rescale = load_video(videopath, every_n_frames=every_n_frames, to_rgb=True,
                                            rescale=facedetection_downsample, inc_pil=True, max_frames=max_frames_to_load)
    if len(pil_frames):
        try:
            faces, coords = detect_face_with_mtcnn(mtcnn_model=mtcnn_model,
                                                   pil_frames=pil_frames,
                                                   facedetection_upsample=1/rescale,
                                                   video_shape=video.shape,
                                                   face_frames=every_n_frames)
        except RuntimeError:  # Out of CUDA RAM
            print(f"Failed to process {videopath} ! Downsampling x2 ...")
            video, pil_frames, rescale = load_video(videopath, every_n_frames=every_n_frames, to_rgb=True,
                                                    rescale=facedetection_downsample/2, inc_pil=True, max_frames=max_frames_to_load)

            try:
                faces, coords = detect_face_with_mtcnn(mtcnn_model=mtcnn_model,
                                                       pil_frames=pil_frames,
                                                       facedetection_upsample=1/rescale,
                                                       video_shape=video.shape,
                                                       face_frames=every_n_frames)
            except RuntimeError:
                print(f"Failed on downsample ! Skipping...")
                return [], []

    else:
        print('Failed to fetch frames ! Skipping ...')
        return [], []

    if len(faces) == 0:
        print('Failed to find faces ! Upsampling x2 ...')
        try:
            video, pil_frames, rescale = load_video(videopath, every_n_frames=every_n_frames, to_rgb=True,
                                                    rescale=facedetection_downsample*2, inc_pil=True, max_frames=max_frames_to_load)
            faces, coords = detect_face_with_mtcnn(mtcnn_model=mtcnn_model,
                                                   pil_frames=pil_frames,
                                                   facedetection_upsample=1/rescale,
                                                   video_shape=video.shape,
                                                   face_frames=every_n_frames)
        except Exception as e:
            print(e)
            return [], []

    return faces, coords


def get_last_frame_needed_across_faces(faces):
    last_frame = 0

    for face in faces:
        (frame_from, frame_to), (row_from, row_to), (col_from, col_to) = face
        last_frame = max(frame_to, last_frame)

    return last_frame


def resize_and_square_face(video, output_size):
    # We will square it, so this is the effective input size
    input_size = max(video.shape[1], video.shape[2])
    out_video = np.empty(
        (len(video), output_size[0], output_size[1], 3), np.dtype('uint8'))

    for i_frame, frame in enumerate(video):
        padded_image = np.zeros((input_size, input_size, 3))
        padded_image[0:frame.shape[0], 0:frame.shape[1]] = frame
        if (input_size, input_size) != output_size:
            frame = cv2.resize(
                padded_image, (output_size[0], output_size[1])).astype(np.uint8)
        else:
            frame = padded_image.astype(np.uint8)
        out_video[i_frame] = frame
    return out_video


def center_crop_video(video, crop_dimensions):
    height, width = video.shape[1], video.shape[2]
    crop_height, crop_width = crop_dimensions

    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width

    video_out = np.zeros((len(video), crop_height, crop_width, 3))
    for i_frame, frame in enumerate(video):
        video_out[i_frame] = frame[y1:y2, x1:x2]

    return video_out
