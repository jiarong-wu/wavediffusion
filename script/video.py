import cv2
import glob
import re

images = glob.glob("sample*.png")

images = sorted(
    images,
    key=lambda s: int(re.search(r"\d+", s).group())
)

# Read first image to get size
frame = cv2.imread(images[0])
h, w, _ = frame.shape

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("video.mp4", fourcc, 10, (w, h))

for img_path in images:
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
