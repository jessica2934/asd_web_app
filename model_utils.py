import pytesseract
import cv2
import numpy as np
import pandas as pd
import re
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
# Cleaning numbers

def clean_number(raw):
    if raw is None:
        return None

    raw = raw.replace(",", ".")
    raw = re.sub(r"[^\d\.]", "", raw)

    if raw == "":
        return None

    if "." not in raw and len(raw) == 3:
        raw = raw[0] + "." + raw[1:]

    try:
        return float(raw)
    except:
        return None


# Extract value

def extract_value(keyword, text):

    lines = text.split("\n")

    for i, line in enumerate(lines):

        if keyword.lower() in line.lower():

            numbers = re.findall(r"[\d\.]+", line)

            if numbers:
                return clean_number(numbers[0])

            for j in range(1,4):

                if i+j < len(lines):

                    numbers = re.findall(r"[\d\.]+", lines[i+j])

                    if numbers:
                        return clean_number(numbers[0])

    return None


# Section splitting

def get_section(text,start_keyword,end_keyword=None):

    start = text.lower().find(start_keyword.lower())

    if start == -1:
        return ""

    if end_keyword:
        end = text.lower().find(end_keyword.lower(),start)
        if end != -1:
            return text[start:end]

    return text[start:]


# OCR + Feature extraction 

def extract_features_from_image(image):

    image = np.array(image)

    gray = cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray,None,fx=2,fy=2,interpolation=cv2.INTER_CUBIC)

    text = pytesseract.image_to_string(gray)

    body_section = get_section(text,"Body composition analysis","Muscle fat analysis")
    obesity_section = get_section(text,"Obesity assessment","Other indicators")
    other_section = get_section(text,"Other indicators")

    data = {}

    data["weight"] = extract_value("Weight",body_section)
    data["body_fat"] = extract_value("Body fat",body_section)
    data["inorganic_salt"] = extract_value("Inorganic",body_section)
    data["protein"] = extract_value("Protein",body_section)
    data["body_water"] = extract_value("Body water",body_section)
    data["muscle_mass"] = extract_value("Muscle",body_section)
    data["skeletal_muscle"] = extract_value("Skeletal",body_section)

    data["BMI"] = extract_value("BMI",obesity_section)
    data["body_fat_rate"] = extract_value("Body fat rate",obesity_section)

    data["visceral_fat"] = extract_value("Visceral",other_section)
    data["BMR"] = extract_value("Basal",other_section)
    data["fat_free_mass"] = extract_value("Fat-free",other_section)
    data["subcutaneous_fat"] = extract_value("Subcutaneous",other_section)
    data["SMI"] = extract_value("SMI",other_section)
    data["WHR"] = extract_value("WHR",other_section)

    return data