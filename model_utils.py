import pytesseract
import cv2
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def clean_number(raw):
    if raw is None:
        return None

    raw = raw.replace(",", ".")
    raw = re.sub(r"[^\d\.]", "", raw)

    if raw == "":
        return None

    if "." not in raw and len(raw) == 3:
        raw = raw[0] + "." + raw[1:]

    parts = raw.split(".")
    if len(parts) > 2:
        raw = parts[0] + "." + "".join(parts[1:])

    try:
        return float(raw)
    except ValueError:
        return None


def extract_value(keyword, text, exclude_keywords=None):
    if exclude_keywords is None:
        exclude_keywords = []

    lines = text.split("\n")

    for i, line in enumerate(lines):

        if keyword.lower() in line.lower():

            skip = False
            for exclude in exclude_keywords:
                if exclude.lower() in line.lower():
                    skip = True
                    break
            if skip:
                continue

            numbers = re.findall(r"[\d\.]+", line)
            if numbers:
                return clean_number(numbers[0])

            for j in range(1, 4):
                if i + j < len(lines):
                    numbers = re.findall(
                        r"[\d\.]+", lines[i + j]
                    )
                    if numbers:
                        return clean_number(numbers[0])

    return None


def get_section(text, start_keyword, end_keyword=None):
    start = text.lower().find(start_keyword.lower())

    if start == -1:
        return ""

    if end_keyword:
        end = text.lower().find(end_keyword.lower(), start + 1)
        if end != -1:
            return text[start:end]

    return text[start:]


def validate_and_fix(data):

    # === WHR: must be 0.5-1.5 ===
    if data["WHR"] is not None:
        if data["WHR"] > 2.0:
            for divisor in [100, 1000, 10]:
                candidate = data["WHR"] / divisor
                if 0.5 <= candidate <= 1.5:
                    print(f"   → WHR fixed: {data['WHR']} "
                          f"-> {round(candidate, 2)}")
                    data["WHR"] = round(candidate, 2)
                    break
            else:
                print(f" WHR {data['WHR']} unreasonable "
                      f"— setting to None")
                data["WHR"] = None
        elif data["WHR"] < 0.3:
            print(f"WHR {data['WHR']} too low "
                  f"— setting to None")
            data["WHR"] = None

    # === BMI: must be 10-60 ===
    if data["BMI"] is not None:
        if data["BMI"] < 10:
            candidate = data["BMI"] * 10
            if 10 <= candidate <= 60:
                print(f"   → BMI fixed: {data['BMI']} "
                      f"-> {round(candidate, 1)}")
                data["BMI"] = round(candidate, 1)
            else:
                print(f"BMI {data['BMI']} unreasonable "
                      f"— setting to None")
                data["BMI"] = None
        elif data["BMI"] > 60:
            print(f"BMI {data['BMI']} unreasonable "
                  f"— setting to None")
            data["BMI"] = None

    # === BMR: must be 500-3000 kcal ===
    if data["BMR"] is not None:
        if data["BMR"] < 100:
            for multiplier in [100, 1000, 10]:
                candidate = data["BMR"] * multiplier
                if 500 <= candidate <= 3000:
                    print(f"   → BMR fixed: {data['BMR']} "
                          f"-> {round(candidate, 0)}")
                    data["BMR"] = round(candidate, 0)
                    break
            else:
                print(f"BMR {data['BMR']} unreasonable "
                      f"— setting to None")
                data["BMR"] = None
        elif data["BMR"] > 3000:
            print(f"BMR {data['BMR']} unreasonable "
                  f"— setting to None")
            data["BMR"] = None

    # === Visceral fat grade: must be 1-30 ===
    if data["visceral_fat"] is not None:
        if data["visceral_fat"] > 30:
            print(f"Visceral fat {data['visceral_fat']} "
                  f"too high — setting to None")
            data["visceral_fat"] = None
        elif data["visceral_fat"] < 1:
            print(f"Visceral fat {data['visceral_fat']} "
                  f"too low — setting to None")
            data["visceral_fat"] = None

    # === Body fat rate: must be 3-60% ===
    if data["body_fat_rate"] is not None:
        if data["body_fat_rate"] > 60:
            # Try dividing by 10 (e.g., 77.0 -> 7.7)
            candidate = data["body_fat_rate"] / 10
            if 3 <= candidate <= 60:
                print(f"   → Body fat rate fixed: "
                      f"{data['body_fat_rate']} "
                      f"-> {round(candidate, 1)}")
                data["body_fat_rate"] = round(candidate, 1)
            else:
                print(f" Body fat rate "
                      f"{data['body_fat_rate']} unreasonable "
                      f"— setting to None")
                data["body_fat_rate"] = None
        elif data["body_fat_rate"] < 3:
            print(f"Body fat rate "
                  f"{data['body_fat_rate']} too low "
                  f"— setting to None")
            data["body_fat_rate"] = None

    # === Weight: must be 15-300 kg ===
    if data["weight"] is not None:
        if data["weight"] < 15 or data["weight"] > 300:
            print(f"Weight {data['weight']} unreasonable "
                  f"— setting to None")
            data["weight"] = None

    # === Body fat (kg) must be less than weight ===
    if (data["body_fat"] is not None and
            data["weight"] is not None):
        if data["body_fat"] > data["weight"]:
            print(f"Body fat {data['body_fat']}kg > "
                  f"Weight {data['weight']}kg "
                  f"— setting body_fat to None")
            data["body_fat"] = None

    # === Inorganic salt: typically 1-6 kg ===
    if data["inorganic_salt"] is not None:
        if data["inorganic_salt"] > 10:
            candidate = data["inorganic_salt"] / 10
            if 1 <= candidate <= 6:
                print(f"   → Inorganic salt fixed: "
                      f"{data['inorganic_salt']} "
                      f"-> {round(candidate, 1)}")
                data["inorganic_salt"] = round(candidate, 1)
            else:
                print(f"Inorganic salt "
                      f"{data['inorganic_salt']} unreasonable "
                      f"— setting to None")
                data["inorganic_salt"] = None

    # === Muscle mass: must be > 5 kg for any person ===
    if data["muscle_mass"] is not None:
        if data["muscle_mass"] < 5:
            print(f"Muscle mass {data['muscle_mass']} "
                  f"too low — setting to None")
            data["muscle_mass"] = None

    # === Fat-free mass: typically 15-100 kg ===
    if data["fat_free_mass"] is not None:
        if data["fat_free_mass"] < 10:
            # Common error: "54.7" read as "5.47"
            candidate = data["fat_free_mass"] * 10
            if 15 <= candidate <= 100:
                print(f"   → Fat-free mass fixed: "
                      f"{data['fat_free_mass']} "
                      f"-> {round(candidate, 1)}")
                data["fat_free_mass"] = round(candidate, 1)
            else:
                print(f"Fat-free mass "
                      f"{data['fat_free_mass']} unreasonable "
                      f"— setting to None")
                data["fat_free_mass"] = None

    return data


def extract_features_from_image(image):
    image = np.array(image)

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.resize(
        gray, None, fx=2, fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    text = pytesseract.image_to_string(gray)

    # Split into sections
    body_section = get_section(
        text,
        "Body composition analysis",
        "Muscle fat analysis"
    )
    obesity_section = get_section(
        text,
        "Obesity assessment",
        "Other indicators"
    )
    other_section = get_section(
        text,
        "Other indicators"
    )

    # === BODY COMPOSITION TABLE ===
    data = {}

    data["weight"] = extract_value(
        "Weight", body_section,
        exclude_keywords=["Weight control"]
    )
    data["body_fat"] = extract_value(
        "Body fat", body_section,
        exclude_keywords=["Body fat rate", "fat rate"]
    )
    data["inorganic_salt"] = extract_value(
        "Inorganic", body_section
    )
    data["protein"] = extract_value(
        "Protein", body_section
    )
    data["body_water"] = extract_value(
        "Body water", body_section
    )
    data["muscle_mass"] = extract_value(
        "Muscle", body_section,
        exclude_keywords=["Skeletal muscle", "Skeletal"]
    )
    data["skeletal_muscle"] = extract_value(
        "Skeletal", body_section
    )

    # === OBESITY ASSESSMENT ===
    data["BMI"] = extract_value(
        "BMI", obesity_section
    )
    data["body_fat_rate"] = extract_value(
        "Body fat rate", obesity_section
    )

    # === OTHER INDICATORS ===
    data["visceral_fat"] = extract_value(
        "Visceral", other_section,
        exclude_keywords=["Basal", "metabolic"]
    )
    data["BMR"] = extract_value(
        "Basal", other_section,
        exclude_keywords=["Visceral"]
    )
    data["fat_free_mass"] = extract_value(
        "Fat-free", other_section
    )
    data["subcutaneous_fat"] = extract_value(
        "Subcutaneous", other_section
    )
    data["SMI"] = extract_value(
        "SMI", other_section
    )
    data["WHR"] = extract_value(
        "WHR", other_section
    )

    # === VALIDATE AND AUTO-FIX ===
    data = validate_and_fix(data)

    # === PRINT RESULTS ===
    print("\n--- EXTRACTED FEATURES ---")
    for key, val in data.items():
        status = "✅" if val is not None else "❌ MISSING"
        print(f"  {key}: {val}  {status}")
    print("--- END FEATURES ---\n")

    return data