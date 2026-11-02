# EasyAnnotate

A lightweight AI-assisted image annotation platform built with Python and Streamlit.

EasyAnnotate uses a pretrained YOLO11n object detection model to automatically generate bounding boxes for road objects. Users can review, edit, add, or remove annotations and export the results in YOLO format.

The project was developed as an individual examination project for the **AI Vision** course at SRH University.

## Features

- **Image Upload:** Upload JPG and PNG images.
- **Manual Annotation:** Draw and edit bounding boxes.
- **AI-Assisted Annotation:** Automatically detect road objects using a pretrained YOLO11n model.
- **Annotation Editing:** Add, resize, move, delete, and change the class of bounding boxes.
- **Save and Restore:** Save annotations locally and restore them when reopening an image.
- **Export:** Download annotated images and YOLO-format labels as a ZIP archive.

## Technology Stack

- Python 3.12
- Streamlit
- Ultralytics YOLO11n
- PyTorch
- Pillow
- Streamlit Drawable Canvas
- Loguru
- uv

The application runs locally using CPU inference. No GPU, model training, or Docker setup is required.

## Project Structure

```text
easy-annotate/
├── app.py
├── src/
│   └── easy_annotate/
│       ├── __init__.py
│       ├── detector.py
│       ├── storage.py
│       └── exporter.py
├── scripts/
│   ├── download_model.py
│   ├── download_dataset.py
│   ├── show_annotation.py
│   └── evaluate.py
├── models/
├── datasets/
│   └── bdd100k-road/
├── data/
│   ├── images/
│   └── annotations/
├── results/
├── pyproject.toml
├── uv.lock
├── requirements.txt
└── README.md
```

## Installation

### Prerequisites

- Python 3.12
- [uv](https://docs.astral.sh/uv/)

Clone the repository:

```bash
git clone https://github.com/SakigamiYang/easy-annotate.git
cd easy-annotate
```

Install dependencies:

```bash
uv sync
```

### Download the Pretrained Model

```bash
uv run python scripts/download_model.py
```

The pretrained model will be saved to:

```text
models/bdd100k-yolo11n.pt
```

### Run the Application

```bash
uv run streamlit run app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

## Usage

1. **Upload an image** using the sidebar.
2. **Annotate manually** by drawing bounding boxes, or click **Auto Annotate** to generate predictions using YOLO11n.
3. **Review and correct** the annotations by modifying bounding boxes, changing classes, adding missing objects, or deleting incorrect detections.
4. Click **Save Annotations** to save the current work locally.
5. Click **Export ZIP** to download the image and annotations in YOLO format.

### Supported Classes

The model detects 10 road-related object categories:

| ID | Class |
|---|---|
| 0 | person |
| 1 | rider |
| 2 | car |
| 3 | truck |
| 4 | bus |
| 5 | train |
| 6 | motor |
| 7 | bike |
| 8 | traffic light |
| 9 | traffic sign |

## Model and Dataset

### Pretrained Model

The application uses a YOLO11n model pretrained on BDD100K.

Source: [dronefreak/bdd100k-yolo11n](https://huggingface.co/dronefreak/bdd100k-yolo11n)

The model is used directly without additional training or fine-tuning.

### Evaluation Dataset

Model quality is evaluated using 100 images from the BDD100K validation dataset, with their original human-annotated ground-truth bounding boxes.

Dataset sources:

- [BDD100K Official Website](https://bdd-data.berkeley.edu/)
- [BDD100K Hugging Face Dataset](https://huggingface.co/datasets/dronefreak/BDD100K)

Download the evaluation samples:

```bash
uv run python scripts/download_dataset.py --count 100 --seed 42
```

The downloaded images, original annotations, and converted YOLO labels are stored in:

```text
datasets/bdd100k-road/
```

### Visualize Ground Truth

Display annotations for a selected image:

```bash
uv run python scripts/show_annotation.py 7
```

Compare human annotations with model predictions:

```bash
uv run python scripts/show_annotation.py 7 --compare
```

## Model Evaluation

Run the evaluation:

```bash
uv run python scripts/evaluate.py
```

The evaluation uses the original human annotations as ground truth and measures object detection quality using standard metrics.

### Evaluation Results

| Metric | Result |
|---|---:|
| Precision | 64.32% |
| Recall | 40.23% |
| mAP@0.5 | 49.34% |
| mAP@0.5:0.95 | 30.16% |

**Evaluation environment:**

- Dataset: BDD100K validation subset
- Images: 100
- Ground-truth instances: 1,744
- Model: YOLO11n, 2,584,102 parameters
- Hardware: Apple M3 CPU
- Python: 3.12.13
- Ultralytics: 8.4.189
- Average inference time: 15.2 ms/image

The results indicate that the model can generate useful initial annotations for road objects, particularly cars and trucks. However, its relatively low recall means that human review remains important for identifying missed objects.

These results are based on a small validation subset and should not be interpreted as a comprehensive benchmark.

## Annotation Export

Annotations can be exported as a ZIP archive containing an image, its YOLO-format annotations, and a dataset configuration file.

Example:

```text
annotations.zip
├── images/
│   └── road.png
├── labels/
│   └── road.txt
└── data.yaml
```

Each annotation is stored as:

```text
class_id x_center y_center width height
```

All bounding-box coordinates are normalized to the range `[0, 1]`.

## Limitations

- The application supports bounding-box annotation only.
- Object categories are limited to the 10 classes supported by the pretrained model.
- Detection accuracy depends on image conditions, object size, and occlusion.
- AI-generated annotations require human verification and correction.
- The model evaluation uses a limited subset of BDD100K images.
- The application is designed as a local single-user demonstration.
