# Apple Grading System using Deep Learning

## Project Overview

This project implements an advanced machine learning solution for automated apple grading using computer vision and deep learning techniques. The system classifies apples into different grades based on their visual characteristics, providing an efficient and accurate method for apple quality assessment.

## 🍎 Features

- **Multi-Class Classification**: Grades apples into four categories
- **Transfer Learning**: Utilizes MobileNetV2 pre-trained model
- **Data Augmentation**: Improves model robustness
- **Visualization**: Provides accuracy plots and confusion matrix
- **Prediction Function**: Allows grading of individual apple images

## 📊 Apple Grades

The system classifies apples into the following grades:
- Apple Green
- Apple Red G1
- Apple Red G2
- Apple Yellow

## 🛠 Technologies Used

- Python
- TensorFlow
- Keras
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn

## 📦 Dataset Structure

```
AppleGradingData/
│
├── Training/
│   ├── Apple Green/
│   ├── Apple Red G1/
│   ├── Apple Red G2/
│   └── Apple Yellow/
│
└── Testing/
    ├── Apple Green/
    ├── Apple Red G1/
    ├── Apple Red G2/
    └── Apple Yellow/
```

## 🚀 Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/apple-grading-system.git
cd apple-grading-system
```

2. Install required dependencies
```bash
pip install -r requirements.txt
```

## 🔧 Usage

### Training the Model
```python
# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=test_generator
)
```

### Predicting Apple Grade
```python
# Predict grade for a single apple image
result = predict_apple_grade('/path/to/apple/image.jpg')
print(f"Grade: {result['grade']}")
print(f"Confidence: {result['confidence']}%")
```

## 📈 Model Performance

### Evaluation Metrics
- **Accuracy**: Model's performance measured on test dataset
- **Confusion Matrix**: Visualizes classification performance
- **Classification Report**: Detailed precision, recall, and F1-score

## 🖼️ Visualizations

The project generates:
- Training History Plot
- Confusion Matrix
- Individual Image Prediction Visualization

## 🔬 Methodology

1. **Data Preprocessing**
   - Image resizing
   - Normalization
   - Data augmentation

2. **Model Architecture**
   - Base Model: MobileNetV2
   - Custom Classification Layers
   - Transfer Learning

3. **Training Strategy**
   - Adam Optimizer
   - Categorical Crossentropy Loss
   - Learning Rate Scheduling

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📞 Contact

Your Name - jayeshwankhede28@gmail.com

## 🙏 Acknowledgements

- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [MobileNetV2](https://arxiv.org/abs/1801.04381)
- Dataset Contributors
