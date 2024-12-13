# Plant Leaf Disease Detection

This repository implements a **Deep Learning Framework** for detecting and classifying plant leaf diseases using images. The system is designed to assist farmers and agricultural experts in identifying and managing plant diseases effectively.

---

## Objectives

1. Detect plant leaf diseases using advanced image classification techniques.
2. Classify leaf diseases into multiple categories based on severity and type.
3. Provide actionable insights for disease management and prevention.

---

## Datasets

### **PlantVillage Dataset**
- **Source**: [PlantVillage Dataset](https://github.com/awslabs/plant-disease-detection)
- **Description**: The dataset contains labeled images of healthy and diseased leaves from various crops.
- **Classes**: Includes 38 classes of plant leaves, including common diseases like rust, mildew, and blight.

---

## Prerequisites

### Tools and Libraries
- Python 3.8+
- TensorFlow / PyTorch
- OpenCV
- Pandas
- NumPy
- Matplotlib

### Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/plant-leaf-disease-detection.git
    cd plant-leaf-disease-detection
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the dataset:
    - Use the [PlantVillage Dataset](https://github.com/awslabs/plant-disease-detection).
    - Place the dataset in the `data/` directory:
      ```
      data/
      ├── train/
      ├── test/
      ├── validation/
      ```

---

## Project Structure

```plaintext
├── data/
│   ├── train/
│   ├── test/
│   ├── validation/
├── models/
│   ├── leaf_disease_model.h5
├── notebooks/
│   ├── training.ipynb
│   ├── testing.ipynb
├── src/
│   ├── train_model.py
│   ├── test_model.py
│   ├── predict.py
├── README.md
├── requirements.txt
```

---

## Usage

### Training the Model
To train the model on the PlantVillage dataset:
```bash
python src/train_model.py --data_dir data/train/ --output_dir models/
```

### Testing the Model
To evaluate the model on the test set:
```bash
python src/test_model.py --data_dir data/test/ --model_path models/leaf_disease_model.h5
```

### Prediction
To predict the disease in a new leaf image:
```bash
python src/predict.py --image_path /path/to/leaf_image.jpg --model_path models/leaf_disease_model.h5
```

---

## Results

- **Training Accuracy**: Achieved XX% accuracy on the training dataset.
- **Testing Accuracy**: Achieved XX% accuracy on the testing dataset.
- **Prediction Time**: Average prediction time per image: XX ms.

---

## Future Work

1. Expand the model to support more crop species and diseases.
2. Integrate the system with mobile and web applications for real-time predictions.
3. Improve model accuracy using advanced architectures like EfficientNet or Vision Transformers.

---
## Acknowledgments

- **PlantVillage Dataset** for providing a comprehensive dataset for plant leaf diseases.
- Mentors and collaborators for their guidance and support.

---
