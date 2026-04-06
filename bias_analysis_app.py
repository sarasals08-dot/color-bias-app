import math
from collections import Counter
from io import BytesIO

import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Color Bias Analysis App", layout="wide")

def normalize_hex(value: str) -> str:
    raw = (value or "").strip().replace("#", "")
    if len(raw) == 3 and all(c in "0123456789abcdefABCDEF" for c in raw):
        raw = "".join(ch * 2 for ch in raw)
    if len(raw) == 8 and all(c in "0123456789abcdefABCDEF" for c in raw):
        raw = raw[:6]
    if len(raw) == 6 and all(c in "0123456789abcdefABCDEF" for c in raw):
        return f"#{raw.upper()}"
    return ""

def hex_to_rgb(hex_value: str):
    normalized = normalize_hex(hex_value)
    if not normalized:
        return None
    raw = normalized[1:]
    return tuple(int(raw[i:i+2], 16) for i in (0, 2, 4))

def rgb_to_hex(rgb):
    r, g, b = [max(0, min(255, int(round(x)))) for x in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"

def rgb_distance(a, b) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def rgb_to_hsl(r, g, b):
    r, g, b = [x / 255 for x in (r, g, b)]
    max_c = max(r, g, b)
    min_c = min(r, g, b)
    l = (max_c + min_c) / 2

    if max_c == min_c:
        h = s = 0.0
    else:
        d = max_c - min_c
        s = d / (2 - max_c - min_c) if l > 0.5 else d / (max_c + min_c)
        if max_c == r:
            h = (g - b) / d + (6 if g < b else 0)
        elif max_c == g:
            h = (b - r) / d + 2
        else:
            h = (r - g) / d + 4
        h /= 6
    return h * 360, s * 100, l * 100

def prepare_image(image: Image.Image, max_side: int = 300):
    image = image.convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(new_size)

@st.cache_data(show_spinner=False)
def analyze_image_bytes(image_bytes: bytes, bucket_size: int = 16):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    image = prepare_image(image)
    arr = np.array(image)
    pixels = arr.reshape(-1, 3)

    avg_rgb = pixels.mean(axis=0)
    average_hex = rgb_to_hex(avg_rgb)

    quantized = (pixels // bucket_size) * bucket_size
    tuples = [tuple(map(int, row)) for row in quantized]
    counts = Counter(tuples)
    total = len(tuples)

    dominant = []
    for color, count in counts.most_common(3):
        dominant.append({
            "hex": rgb_to_hex(color),
            "count": count,
            "share": round(count / total * 100, 2),
        })

    return {
        "average_hex": average_hex,
        "dominant_colors": dominant,
        "pixel_count": total,
        "resized_dimensions": image.size,
    }

def average_pairwise_distance(colors):
    if len(colors) < 2:
        return 0.0
    distances = []
    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            distances.append(rgb_distance(colors[i], colors[j]))
    return float(sum(distances) / len(distances)) if distances else 0.0

def generate_bias_analysis(user_hexes, dominant_hexes, average_hex):
    user_rgbs = [hex_to_rgb(h) for h in user_hexes if normalize_hex(h)]
    dominant_rgbs = [hex_to_rgb(h) for h in dominant_hexes if normalize_hex(h)]
    avg_rgb = hex_to_rgb(average_hex)

    if not user_rgbs or not dominant_rgbs or not avg_rgb:
        return [{"title": "Insufficient data",
                 "text": "Enter at least one valid swatch and analyze an image."}]

    insights = []

    user_lightness = np.mean([rgb_to_hsl(*rgb)[2] for rgb in user_rgbs])
    dominant_lightness = np.mean([rgb_to_hsl(*rgb)[2] for rgb in dominant_rgbs])
    if user_lightness > dominant_lightness + 8:
        insights.append({"title": "Brightness bias",
                         "text": "You selected brighter colors than the dominant pixels."})

    user_spread = average_pairwise_distance(user_rgbs)
    dominant_spread = average_pairwise_distance(dominant_rgbs)
    if user_spread > dominant_spread + 20:
        insights.append({"title": "Contrast bias",
                         "text": "Your colors are more different from each other."})

    avg_distance = np.mean([rgb_distance(c, avg_rgb) for c in user_rgbs])
    if avg_distance > 70:
        insights.append({"title": "Saliency bias",
                         "text": "You chose attention-grabbing colors."})

    if not insights:
        insights.append({"title": "Close alignment",
                         "text": "Your choices match the image statistics fairly well."})

    return insights

st.title("Color Bias Analysis App")

uploaded_files = st.file_uploader(
    "Upload images", type=["png", "jpg", "jpeg"], accept_multiple_files=True
)

if not uploaded_files:
    st.info("Upload at least one image.")
    st.stop()

for idx, file in enumerate(uploaded_files, start=1):
    st.subheader(f"Image {idx}: {file.name}")
    image = Image.open(file)
    st.image(image)

    c1, c2, c3 = st.columns(3)
    with c1:
        h1 = st.text_input("Color 1", key=f"h1_{idx}")
    with c2:
        h2 = st.text_input("Color 2", key=f"h2_{idx}")
    with c3:
        h3 = st.text_input("Color 3", key=f"h3_{idx}")

    if st.button("Analyze", key=f"btn_{idx}"):
        result = analyze_image_bytes(file.getvalue())
        dom = [c["hex"] for c in result["dominant_colors"]]

        st.write("Average color:", result["average_hex"])
        st.write("Dominant colors:", dom)

        insights = generate_bias_analysis([h1, h2, h3], dom, result["average_hex"])
        st.write("Bias Analysis:")
        for ins in insights:
            st.write("-", ins["title"], ":", ins["text"])
