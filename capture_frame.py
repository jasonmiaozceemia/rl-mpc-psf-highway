import os
import imageio
import matplotlib.pyplot as plt

def show_key_frames_individually(video_path, frames_of_interest):
    """
    Load and display key frames individually from the video at video_path.
    frames_of_interest is a list of frame indices to display.
    """
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        return

    reader = imageio.get_reader(video_path, 'ffmpeg')
    total_frames = reader.get_length()

    for idx in frames_of_interest:
        if idx < total_frames:
            frame = reader.get_data(idx)
            plt.figure()
            plt.imshow(frame)
            plt.title(f"Frame {idx}")
            plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Requested frame {idx} exceeds total frame count ({total_frames}). Skipping.")

if __name__ == "__main__":
    # Specify the path to your video file
    video_folder = "video_mpc"
    video_filename = "casadi_mpc_run-episode-0.mp4"
    video_path = os.path.join(video_folder, video_filename)

    # Specify key frame indices to display
    frames_of_interest = [0,4,6,12,16]

    show_key_frames_individually(video_path, frames_of_interest)
