# Realtime Skin Problems Detection with YOLOv5  

This project implements real-time skin problem detection using **YOLOv5** and a Flask-based web application. The model can detect skin conditions from images, videos, or live webcam feeds.  

## Features  

- **Real-time detection**: Supports image and video input, including live webcam feeds.  
- **Flask-based web app**: Simple web interface to upload images or videos for analysis.  
- **YOLOv5-based detection**: Custom-trained YOLOv5 model for skin problem classification.  

## Project Structure  

```
Video_Frame_Split/  
│── model/                   # Pretrained YOLOv5 models (best.pt, best8.pt, etc.)  
│── static/                  # Static files (cover image, PyTorch logo, styles)  
|── templates/index.html      # Frontend (HTML template for Flask app)  
|── main.py                   # Main Flask application script  
│── yolov5s.pt                # YOLOv5 model (this is not the current model used in main.py)  
│── README.md                 # Documentation  
```

## Installation  

1. **Clone the repository**  
   ```sh
   git clone https://github.com/ReinerJasin/Realtime-Skin-Problems-Detection-with-YOLO-V5.git
   cd Realtime-Skin-Problems-Detection-with-YOLO-V5
   ```

2. **Install dependencies** (Create a `requirements.txt` if missing)  
   ```sh
   pip install torch torchvision torchaudio flask opencv-python pillow
   ```

3. **Download YOLOv5 repository**  
   ```sh
   git clone https://github.com/ultralytics/yolov5.git
   cd yolov5
   
   pip install -r requirements.txt
   cd ..
   ```

## Usage  

### Run the Flask App  
```sh
python main.py
```
Then, open `http://127.0.0.1:5000/` in your browser.  

### Upload Images or Videos  
- Click on the upload button to submit an image or video for processing.  
- The model processes and displays results directly in the browser.  

### Webcam Detection  
- Visit `http://127.0.0.1:5000/video_feed` to see real-time predictions.  

## Model Details  

- The model is loaded using:  
  ```python
  model = torch.hub.load('ultralytics/yolov5', 'custom', path="model/best.pt", force_reload=True)
  ```
- `best.pt` is the custom-trained YOLOv5 model for skin problem detection.  

## License  

This project is licensed under the **Apache License 2.0**.  

You are free to use, modify, and distribute this project, but **attribution is required**. Please provide proper credit by linking to this repository when using the project or its components.  

For more details, see the [LICENSE](https://github.com/ReinerJasin/Realtime-Skin-Problems-Detection-with-YOLO-V5/blob/main/LICENSE.txt) file.  

---
