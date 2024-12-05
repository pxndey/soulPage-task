import easyocr
import ultralytics
import numpy as np
import pandas as pd
import cv2
import os

model = ultralytics.YOLO("weights/bb_weights.pt")
reader = easyocr.Reader(['en'])

def saveBoundingBox(image):
    """
        Function to save the bounding box of the image into the results directory
    """
    results = model.predict(image,imgsz=640)
    bbox = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, bbox.xyxy[0].tolist())
    cropped = image[y1:y2, x1:x2]
    return cropped


def getOCR(croppedImages):
    """
        Function to read in the cropped boudning boxes and to return the text processed by them
    """
    text = reader.readtext(croppedImages)
    if len(text)==1:
        return text[0][1]
    else:
        newText = ""
        for t in text:
            newText+=t[1]
        return newText

def format_text_output():
    """
        Read images, perform cropping, OCR and format in given format
    """
    output = ["id,0,1,2,3,4,5,6,7,8,9"]
    df_rows = []
    textArr = []
    path = "data/test"
    
    if os.path.isdir(path):
        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)
            image = cv2.imread(file_path)
            croppedBox = saveBoundingBox(image)
            text = getOCR(croppedBox)
            textArr.append(text)
    
            
            counter = 1  
            for char in text:
                if char.isdigit():  
                    row = [f"img_{os.path.splitext(filename)[0]}_{counter}"] + ["0"] * 10
                    row[int(char) + 1] = "1"  
                    df_rows.append(row)
                    counter += 1
    
    # Create the DataFrame and save it as a CSV
    columns = ["id"] + [str(i) for i in range(10)]
    df = pd.DataFrame(df_rows, columns=columns)
    df.to_csv("submission.csv", index=False)



if __name__=="__main__":
    format_text_output()
