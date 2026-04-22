import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
from analyze_emotions import analyze_emotions

# ====================================================
# Page config
# ====================================================
st.set_page_config(page_title="Text Emotion Visualizer", layout="centered")
st.title("Text Emotion Visualizer")

# ====================================================
# Load default text from file
# ====================================================
if "default_text" not in st.session_state:
    with open("relevant_text.txt", "r", encoding="utf-8") as f:
        st.session_state.default_text = f.read()

# ====================================================
# Analyze and display result
# ====================================================
if "emotions" not in st.session_state:
    st.session_state.emotions = None

if st.session_state.emotions:
    emotions = st.session_state.emotions

    # --- Build sigmoid gradient (top-2 emotions) ---
    top_color_hex = emotions[0]["color"]
    bot_color_hex = emotions[1]["color"] if len(emotions) > 1 else emotions[0]["color"]
    top_score = emotions[0]["probability"]

    width, height = 150, 400
    steepness = 10
    transition_height = round(top_score * height)

    image = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)

    top_r, top_g, top_b = int(top_color_hex[1:3], 16), int(top_color_hex[3:5], 16), int(top_color_hex[5:7], 16)
    bot_r, bot_g, bot_b = int(bot_color_hex[1:3], 16), int(bot_color_hex[3:5], 16), int(bot_color_hex[5:7], 16)

    for i in range(height):
        ratio = 1 / (1 + np.exp(-steepness * (i - transition_height) / height))
        r = round((1 - ratio) * top_r + ratio * bot_r)
        g = round((1 - ratio) * top_g + ratio * bot_g)
        b = round((1 - ratio) * top_b + ratio * bot_b)
        draw.line([(0, i), (width, i)], fill=(r, g, b))

    # --- Layout: gradient left, emotions right ---
    col_img, col_text = st.columns([1, 2])

    with col_img:
        st.image(image, use_container_width=False)

    with col_text:
        st.markdown("<br><br>", unsafe_allow_html=True)
        for e in emotions:
            st.markdown(
                f'<p style="color:{e["color"]}; font-size:20px; margin:4px 0;">'
                f'<b>{e["emotion"]}</b> — {e["probability"]:.2f}</p>',
                unsafe_allow_html=True,
            )

    st.divider()

# ====================================================
# Text input and button (below the rectangle)
# ====================================================
user_text = st.text_area("Enter text to analyze:", value=st.session_state.default_text, height=200)

if st.button("Analyze emotions", type="primary"):
    st.session_state.emotions = analyze_emotions(user_text)
    st.rerun()
