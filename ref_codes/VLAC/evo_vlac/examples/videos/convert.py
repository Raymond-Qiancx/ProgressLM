import cv2
import os
# video_path = '/home/vcj9002/jianshu/chengxuan/ProgressLM/ref_codes/VLAC/evo_vlac/examples/videos/pick-bowl-test.mp4'
video_path = '/projects/b1222/userdata/jianshu/chengxuan/ProgressLM//ref_codes/VLAC/evo_vlac/examples/videos/pick-bowl-ref.mov'
output_dir = '/projects/b1222/userdata/jianshu/chengxuan/ProgressLM//ref_codes/VLAC/evo_vlac/examples/videos/ref_images'
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_count = 0
save_interval = 30  # 每隔10帧保存一次，可自行调整

while True:
    ret, frame = cap.read()
    if not ret:
        break
    if frame_count % save_interval == 0:
        frame_name = os.path.join(output_dir, f"frame_{frame_count:04d}.jpg")
        cv2.imwrite(frame_name, frame)
    frame_count += 1

cap.release()
print(f"共读取 {frame_count} 帧，保存了约 {frame_count // save_interval} 张图像。")
