# To background video, adjust its resolution, frame number:
python get_bgvideo.py --bg_video_path "../assets/bg/45861.mp4" --save_path "../assets/bg/source/45861.mp4"

# Get foreground video, get its binary mask of foreground element. We use sam2 with textual instruction here:
python get_fgmask.py --fg_video_path "../assets/fg/45871.mp4" --prompt "bird"