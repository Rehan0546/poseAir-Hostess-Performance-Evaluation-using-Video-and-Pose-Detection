
# Air Hostess Performance Evaluation using Video and Pose Detection

Note: Other files can be downlaoded from yolov5 repository, https://github.com/ultralytics/yolov5
Full code: https://github.com/Rehan0546/pose_estimation_score_airhosetess_rehearsal

This repository contains a Python script designed to evaluate the performance of air hostesses by analyzing video footage. The script detects and tracks individuals, identifies specific actions based on audio cues, and calculates pose accuracy scores by comparing detected poses against predefined reference poses.

## Features

- **Pose Detection and Analysis**: Uses MediaPipe and OpenCV to detect human poses and analyze specific joint angles.
- **Audio-Based Action Classification**: Converts video audio to text, segments it into action classes, and associates them with video frames.
- **Pose Matching and Scoring**: Matches the detected poses of air hostesses with predefined actions and calculates an accuracy score.
- **Face and Smile Detection**: Detects faces and smiles in the video frames to enhance pose analysis.
- **Video Processing and Annotation**: Processes video frames to draw bounding boxes and annotate with pose detection results and scores.
- **Multi-Person Tracking**: Uses YOLOv5 for object detection and SORT for tracking individuals across video frames.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- Torch
- NumPy
- Pandas
- pydub
- SpeechRecognition
- scikit-learn
- MoviePy

Install the required packages using:
```bash
pip install opencv-python mediapipe torch numpy pandas pydub SpeechRecognition scikit-learn moviepy
```

## Usage

1. Place your video file and pose reference CSV files in the appropriate directories.
2. Update the `weights` and `source` variables in the script if needed.
3. Run the script:
    ```bash
    python script_name.py --source path/to/video.mp4 --weights yolov5x.pt --img-size 640 --conf-thres 0.45 --iou-thres 0.45
    ```

## Script Description

The script is organized into several classes and functions:

### `Face_smile_detect`
Detects faces and smiles in images using OpenCV's pre-trained models. Draws bounding boxes around detected faces and smiles.

### `mp4tomp3`
Converts video files (MP4) to audio files (MP3).

### `class_detect`
Analyzes chunks of transcribed audio to classify segments based on predefined keywords.

### `fps_counter`
Calculates start and end frames for each detected class segment based on video FPS.

### `video_audio_to_text`
Converts video files to audio, splits the audio into chunks, and transcribes each chunk using Google Speech Recognition.

### `PoseDetectors`
Uses MediaPipe to detect and analyze human poses in images. Calculates 2D and 3D angles between specific joints.

### `error_find`
Compares detected poses against reference data to calculate error scores for specific actions.

### `plot_one_box`
Draws bounding boxes on images and annotates them with pose detection results and scores.

### `detect`
Main function for processing video frames to detect, track, and analyze people. Uses YOLOv5 for object detection and SORT for tracking. Integrates pose detection and action classification.

## Workflow

1. **Initialization**: Load models and set up configurations.
2. **Video Processing**:
   - Read video frames and detect objects using YOLOv5.
   - Track detected objects across frames using SORT.
   - Perform face and smile detection, as well as pose detection.
   - Calculate and display pose angles and scores.
3. **Action Classification**:
   - Transcribe audio from video to detect and classify specific actions.
   - Calculate start and end frames for each action.
4. **Result Saving**:
   - Save processed video with annotations.
   - Save action scores and pose detection results in a JSON file.

## Execution
The script can be executed from the command line with various options for input source, model weights, image size, confidence thresholds, device selection, and output settings.

Example command:
```bash
python script_name.py --source path/to/video.mp4 --weights yolov5x.pt --img-size 640 --conf-thres 0.45 --iou-thres 0.45
```

This script provides a comprehensive solution for evaluating air hostess performance, combining object detection, tracking, pose estimation, and action recognition to calculate accuracy scores based on predefined reference poses and actions.

## Author

Rehan
