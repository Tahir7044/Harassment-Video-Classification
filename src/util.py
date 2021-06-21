def positive_annotating_points(path):
    file = open(path, "r")
    points = []
    for point in file:
        pt = [int(x) for x in point.split(" ")]
        points.append(pt)
    return points


def get_frame_label(frame_count, annotation_path):
    frame_label = [0]*(frame_count+1)
    pos_frames = positive_annotating_points(annotation_path)
    for frame in pos_frames:
        start, end = frame
        frame_label[start] = 1
        frame_label[end+1] = -1
    for i in range(1, len(frame_label)):
        frame_label[i] += frame_label[i-1]
    return frame_label
