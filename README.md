# Deepfake Image and Video Detection

A deep learning–based system for detecting AI-generated fake images and videos (deepfakes).
This project leverages convolutional neural networks and transfer learning to identify
manipulated visual content by learning subtle artifacts that are not easily detectable
by the human eye.

The system focuses on face-based analysis and supports both image and video inputs,
making it suitable for real-world applications such as media verification,
digital forensics, and misinformation prevention.

---

## Project Motivation

With the rapid advancement of generative AI models, creating realistic fake images
and videos has become easier than ever. These deepfakes pose serious threats in the
form of misinformation, identity misuse, and reputational damage.

Traditional forensic techniques are no longer sufficient to detect such highly
realistic synthetic media. This project aims to address this challenge using
deep learning and transfer learning–based detection techniques.

---

## Objectives

- Detect deepfake images and videos with high accuracy
- Utilize pretrained CNN models for effective feature extraction
- Apply transfer learning to reduce training time and improve performance
- Perform automated face detection and preprocessing
- Ensure generalization across multiple datasets
- Provide a simple interface for real-time prediction

---

## Key Features

- Image and video deepfake detection
- Face region extraction and preprocessing
- Transfer learning using pretrained CNN models
- Binary classification (Real vs Fake)
- Confidence score for predictions
- Scalable and modular project structure
- Suitable for academic, research, and demo purposes

---

## Technology Stack

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Face Recognition / Dlib
- Streamlit or Flask (for UI)
- Google Colab / GPU-enabled system (optional)

---

## Models Used

- Convolutional Neural Network (CNN)
- VGG16
- MobileNet
- ResNet-50

These models are fine-tuned on deepfake datasets to learn manipulation artifacts
introduced during image or video synthesis.

---

## Dataset

- FaceForensics++
- Celeb-DF
- DFDC (DeepFake Detection Challenge)
- Fake vs Real Image datasets

The system is trained on labeled datasets containing both real and manipulated samples
to ensure robust detection.

---

## Methodology

1. Data Collection  
   Collection of real and fake images/videos from publicly available datasets.

2. Preprocessing  
   - Face detection and cropping  
   - Image resizing and normalization  
   - Frame extraction for video inputs  

3. Feature Extraction  
   Use of pretrained CNN models for deep feature learning.

4. Model Training  
   - Binary classification (Real / Fake)  
   - Evaluation using accuracy, precision, recall, and F1-score  

5. Prediction & Visualization  
   Real-time prediction with confidence score through a user interface.

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score

These metrics are used to measure model performance and generalization.

---

## Applications

- Fake media detection
- Digital forensics
- Social media content verification
- Journalism and fact-checking
- Cybersecurity and identity protection

---

## Future Enhancements

- Improve robustness against unseen deepfake techniques
- Add audio-based deepfake detection
- Deploy as a cloud-based API
- Integrate explainable AI (XAI) for better interpretability
- Support real-time video stream detection

---

## Author

Manthan Ghatbandhe  
Bachelor of Technology – Computer Science and Engineering  
Government College of Engineering, Nagpur  
  [LinkedIn](https://www.linkedin.com/in/manthan-ghathbandhe-852945258/) | [GitHub](https://github.com/GhatbandheManthan)
