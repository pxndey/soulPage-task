# soulPage-task

## About
- *bb_det.ipynb* contains training code for a YOLOv8 model fine tuned on the number plate bounding box provided.

- *ocr_det.ipnb* contains testing of text extraction using easyOCR on provided number plates

- *mods.py* contains modular code to:
  - Read images of environment
  - find, and crop the number plate
  - feed cropped number plate to easyOCR
  - extract text and formatting it in the *sampleSubmission.csv* format given

## Setup

- Clone the Repository
```bash
git clone https://github.com/pxndey/soulPage-task
cd soulPage-task
```

- Create and activate the virtual environment
```bash
python -m venv venv
venv/Scripts/activate
```

- Install required Dependencies from the root folder
```bash
pip install -r requirements.txt
```

- Run the python file
```bash
python mods.py
```

- The outputs will be stored in *submission.csv*