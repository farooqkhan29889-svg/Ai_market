import os
import base64
import streamlit as st
import requests
from urllib.parse import quote

# ---- 0. Load the secret key from .env (visitors can't see it) ----
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    # strip spaces AND quotes (some people write "key" with quotes)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

load_env()

# ---- The "brain" for all text jobs. Tries Groq first, then Gemini, silently. ----
def ask_gemini_text(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "no GEMINI_API_KEY in .env"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    r = requests.post(url, params={"key": api_key},
                      json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=90)
    if r.status_code != 200:
        return None, f"Gemini error {r.status_code}: {r.text[:200]}"
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts), None

BUSY_MSG = "Sorry, the AI is busy right now. Please try again in a minute."

def ask_text_ai(prompt):
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            data = {"model": "openai/gpt-oss-120b",
                    "messages": [{"role": "user", "content": prompt}]}
            r = requests.post(url, headers=headers, json=data, timeout=60)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
    import time
    for _ in range(3):
        text, err = ask_gemini_text(prompt)
        if text:
            return text
        time.sleep(2)
    return None

# ---- Ask AI about a picture (like ChatGPT with an image) ----
# Groq no longer has vision models, so Google's Gemini does this job.
def ask_gemini_vision(prompt, img_bytes):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "No GEMINI_API_KEY found in .env file."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    b64 = base64.b64encode(img_bytes).decode()
    data = {"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
    ]}]}
    r = requests.post(url, params={"key": api_key}, json=data, timeout=120)
    if r.status_code != 200:
        return f"AI error {r.status_code}: {r.text[:200]}"
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts) or "AI gave an empty answer."

# ---- The function that asks Google's AI to draw a picture ----
def ask_gemini_image(prompt):
    # The "Nano Banana" family: tries Banana 2 first, then Banana Pro, then the old Banana
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "no key"
    models = ["gemini-nano-banana-2.1", "gemini-3-pro-image-preview",
              "gemini-2.5-flash-image"]
    for model in models:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        r = requests.post(url, params={"key": api_key},
                          json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=180)
        if r.status_code != 200:
            continue
        parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        for part in parts:
            if "inlineData" in part:
                return base64.b64decode(part["inlineData"]["data"]), None
        return None, "No image in the answer"
    return None, f"Gemini error {r.status_code}: {r.text[:200]}"

# ---- Edit an uploaded photo with Gemini (Nano Banana can take a photo in) ----
def ask_gemini_edit_image(prompt, img_bytes):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "no key"
    b64 = base64.b64encode(img_bytes).decode()
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-nano-banana-2.1:generateContent"
    data = {"contents": [{"parts": [
        {"text": prompt},
        {"inline_data": {"mime_type": "image/jpeg", "data": b64}}
    ]}]}
    r = requests.post(url, params={"key": api_key}, json=data, timeout=180)
    if r.status_code == 429:
        return None, "quota"
    if r.status_code != 200:
        return None, f"Gemini error {r.status_code}: {r.text[:200]}"
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for part in parts:
        if "inlineData" in part:
            return base64.b64decode(part["inlineData"]["data"]), None
    return None, "No image in the answer"

# ---- Edit a photo: Cloudflare first, then Gemini as backup ----
def ask_edit_photo(prompt, img_bytes):
    edited, err = ask_cf_edit_image(prompt, img_bytes)
    if not err:
        return edited, None
    cf_err = err
    edited, err = ask_gemini_edit_image(prompt, img_bytes)
    if not err:
        return edited, None
    if cf_err == "daily limit" and err == "quota":
        return None, "daily limit"
    return None, "busy"

# ---- Backup 1: Together AI's free FLUX picture model ----
def ask_together_image(prompt):
    api_key = os.environ.get("TOGETHER_API_KEY")
    if not api_key:
        return None, "no key"
    url = "https://api.together.xyz/v1/images/generations"
    r = requests.post(url, headers={"Authorization": f"Bearer {api_key}"},
                      json={"model": "black-forest-labs/FLUX.1-schnell-Free",
                            "prompt": prompt, "width": 1024, "height": 1024,
                            "response_format": "b64_json"}, timeout=180)
    if r.status_code != 200:
        return None, f"Together error {r.status_code}: {r.text[:200]}"
    b64 = r.json().get("data", [{}])[0].get("b64_json")
    if not b64:
        return None, "No image in the answer"
    return base64.b64decode(b64), None

# ---- Backup 2: Cloudflare Workers AI (free, no card needed) ----
def ask_cf_image(prompt):
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if not account_id or not api_token:
        return None, "no key"
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    r = requests.post(url, headers={"Authorization": f"Bearer {api_token}"},
                      json={"prompt": prompt}, timeout=180)
    if r.status_code != 200:
        return None, f"Cloudflare error {r.status_code}: {r.text[:200]}"
    img_b64 = r.json().get("result", {}).get("image")
    if not img_b64:
        return None, "No image in the answer"
    return base64.b64decode(img_b64), None

# ---- Edit an uploaded photo: FLUX.2 takes your picture + your words ----
def ask_cf_edit_image(prompt, img_bytes):
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if not account_id or not api_token:
        return None, "no key"
    url = (f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
           f"/ai/run/@cf/black-forest-labs/flux-2-dev")
    r = requests.post(url, headers={"Authorization": f"Bearer {api_token}"},
                      files={"input_image_0": ("photo.jpg", img_bytes, "image/jpeg")},
                      data={"prompt": prompt, "steps": "25"}, timeout=(10, 180))
    if r.status_code != 200:
        if "10,000 neurons" in r.text:
            return None, "daily limit"
        return None, f"Cloudflare error {r.status_code}: {r.text[:200]}"
    img_b64 = r.json().get("result", {}).get("image")
    if not img_b64:
        return None, "No image in the answer"
    return base64.b64decode(img_b64), None

# ---- Write words on a photo (no AI, no limit, always works) ----
def write_text_on_photo(img_bytes, words, position="bottom", color="white", size="Medium"):
    import io
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    # Font size scales with the photo width, so text looks right on any photo size
    factors = {"Small": 0.03, "Medium": 0.05, "Big": 0.075}
    px = max(18, int(w * factors.get(size, 0.05)))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", px)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), words, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = (w - tw) // 2 - bbox[0]
    if position == "top":
        y = h // 20 - bbox[1]
    elif position == "middle":
        y = (h - th) // 2 - bbox[1]
    else:
        y = h - th - h // 20 - bbox[1]
    draw.text((x + 3, y + 3), words, font=font, fill="black")
    draw.text((x, y), words, font=font, fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

# ---- Turn the resume text into a real PDF file ----
def make_resume_pdf(text, header=None):
    from fpdf import FPDF
    # The AI sometimes writes "smart" characters (fancy spaces, dashes, quotes).
    replacements = {
        "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ",
        "\u2011": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for smart, normal in replacements.items():
        text = text.replace(smart, normal)

    pdf = FPDF("p", "mm", "A4")
    # Arial (a Unicode font) can print the real bullet dot. On computers without it
    # we fall back to the built-in font and a plain "-" bullet.
    try:
        pdf.add_font("Body", "", "C:/Windows/Fonts/arial.ttf")
        pdf.add_font("Body", "B", "C:/Windows/Fonts/arialbd.ttf")
        font, bullet = "Body", "•"
    except Exception:
        font, bullet = "Helvetica", "-"
        text = text.replace("•", "-").encode("latin-1", errors="replace").decode("latin-1")

    NAVY = (49, 46, 129)
    INK = (30, 30, 30)

    import re
    # The AI sometimes leaves instructions in brackets, like "[University Name]"
    # or "[Year of Completion]". A customer must never see those in their PDF, so
    # remove every bracketed note and drop the lines that had nothing else left.
    text = re.sub(r"\[[^\]]{0,60}\]", "", text)
    lines = []
    for raw in text.split("\n"):
        if raw.strip() and not re.sub(r"^[\s#*\-•|:;,]+", "", raw).strip():
            continue
        lines.append(re.sub(r"\s*[|,;]\s*$", "", raw))

    # The AI repeats the name / email / phone at the top. The header band shows
    # those already, so drop the duplicate lines instead of printing them twice.
    if header:
        keys = [str(header.get(k, "")).strip().lower() for k in ("name", "email", "phone")]
        keys = [k for k in keys if k]
        keys += [p.strip().lower() for p in str(header.get("contact", "")).split("|") if len(p.strip()) > 3]
        while lines:
            first = lines[0].strip()
            if not first:
                lines.pop(0)
                continue
            if len(first) < 90 and any(k in first.lower() for k in keys):
                lines.pop(0)
                continue
            break

    # ---- Group the lines into sections (heading + its bullet lines) ----
    sections, pre = [], []
    for raw in lines:
        line = raw.strip().replace("\t", " ")
        if not line:
            continue
        if line.startswith("#"):
            title = line.lstrip("#").strip().replace("*", "").replace("_", "")
            if title.lower() in ("resume", "curriculum vitae", "cv"):
                continue
            sections.append({"title": title, "body": []})
        elif sections:
            sections[-1]["body"].append(line)
        else:
            pre.append(line)
    if pre:
        sections.insert(0, {"title": "", "body": pre})

    # Short facts go in the narrow left column, the story goes on the right.
    LEFT_KEYS = ("skill", "educat", "personal", "detail", "contact", "language",
                 "certif", "interest", "declaration", "reference", "award", "hobby")
    left_idx = {i for i, s in enumerate(sections)
                if any(k in s["title"].lower() for k in LEFT_KEYS)}
    left_secs = [s for i, s in enumerate(sections) if i in left_idx]
    right_secs = [s for i, s in enumerate(sections) if i not in left_idx]
    if not left_secs:  # nothing to put on the side -> one wide column
        right_secs = sections

    MARGIN, GAP, BODY, LH = 14, 8, 9.5, 4.8
    left_w = 62 if left_secs else 0
    right_w = 210 - 2 * MARGIN - (left_w + GAP if left_secs else 0)

    pdf.set_auto_page_break(auto=False)
    pdf.set_margins(MARGIN, MARGIN, MARGIN)
    pdf.add_page()

    # Dark blue band with the name, the job they want, and contact details
    band_bottom = MARGIN
    if header and header.get("name"):
        band_h = 28 if (header.get("title") or header.get("contact")) else 20
        pdf.set_fill_color(*NAVY)
        pdf.rect(0, 0, 210, band_h, style="F")
        pdf.set_y(6)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font(font, "B", 20)
        pdf.cell(0, 8, header["name"], new_x="LMARGIN", new_y="NEXT", align="C")
        if header.get("title"):
            pdf.set_font(font, "", 11)
            pdf.cell(0, 6, header["title"], new_x="LMARGIN", new_y="NEXT", align="C")
        if header.get("contact"):
            pdf.set_font(font, "", 9)
            pdf.multi_cell(0, 5, header["contact"], new_x="LMARGIN", new_y="NEXT", align="C")
        band_bottom = band_h
    pdf.set_text_color(*INK)

    def measure(sec, col_w):
        """How tall a section will be, so we know when to start the next page."""
        h = 0
        if sec["title"]:
            h += 6 + 3.2
        pdf.set_font(font, "", BODY)
        for line in sec["body"]:
            if line[:2] in ("- ", "* ") or line.startswith("•"):
                h += pdf.multi_cell(col_w - 6, LH, line.lstrip("-*• \t"),
                                    dry_run=True, output="HEIGHT", markdown=True)
            else:
                h += pdf.multi_cell(col_w, LH, line,
                                    dry_run=True, output="HEIGHT", markdown=True)
        return h + 3

    def draw(sec, x, y, col_w):
        if sec["title"]:
            pdf.set_xy(x, y)
            pdf.set_font(font, "B", 10.5)
            pdf.set_text_color(*NAVY)
            pdf.cell(col_w, 6, sec["title"].upper())
            pdf.set_draw_color(*NAVY)
            pdf.set_line_width(0.35)
            pdf.line(x, y + 6.8, x + col_w, y + 6.8)
            y += 6 + 3.2
            pdf.set_text_color(*INK)
        pdf.set_font(font, "", BODY)
        for line in sec["body"]:
            pdf.set_xy(x, y)
            if line[:2] in ("- ", "* ") or line.startswith("•"):
                pdf.cell(6, LH, bullet)
                pdf.multi_cell(col_w - 6, LH, line.lstrip("-*• \t"),
                               new_x="LEFT", new_y="NEXT", markdown=True, align="L")
            else:
                pdf.multi_cell(col_w, LH, line,
                               new_x="LEFT", new_y="NEXT", markdown=True, align="L")
            y = pdf.get_y()

    def pack(secs, col_w, top, bottom):
        """Split sections into pages: returns [[(y, section), ...], ...]."""
        pages, current, y = [], [], top
        for sec in secs:
            h = measure(sec, col_w)
            if current and y + h > bottom:
                pages.append(current)
                current, y = [], MARGIN
            current.append((y, sec))
            y += h
        if current:
            pages.append(current)
        return pages or [[]]

    TOP, BOTTOM = band_bottom + 7, 297 - MARGIN
    left_pages = pack(left_secs, left_w, TOP, BOTTOM) if left_secs else []
    right_pages = pack(right_secs, right_w, TOP, BOTTOM)
    for page in range(max(len(left_pages), len(right_pages))):
        if page:
            pdf.add_page()
        if page < len(right_pages):
            for y, sec in right_pages[page]:
                draw(sec, MARGIN + left_w + (GAP if left_secs else 0), y, right_w)
        if page < len(left_pages):
            for y, sec in left_pages[page]:
                draw(sec, MARGIN, y, left_w)
    return bytes(pdf.output())

# ---- Free AI voice (Microsoft edge-tts, no key needed) ----
def make_voice(text, voice="en-US-JennyNeural", filename="voice.mp3"):
    import asyncio
    import edge_tts
    async def _run():
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(filename)
    loop = asyncio.new_event_loop()
    loop.run_until_complete(_run())
    loop.close()
    return filename

# ---- Turn script text into clean scene lines (works for AI and user-written scripts) ----
def parse_script_lines(text):
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        if ln.startswith("- "):
            ln = ln[2:].strip()
        lines.append(ln)
    return lines

# ---- Picture for one reels scene (Cloudflare first, Gemini as backup) ----
def ask_scene_image(prompt):
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if account_id and api_token:
        url = (f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
               f"/ai/run/@cf/black-forest-labs/flux-1-schnell")
        # 2 tries: the free service sometimes takes a moment and needs a second attempt
        for attempt in range(2):
            try:
                r = requests.post(url, headers={"Authorization": f"Bearer {api_token}"},
                                  json={"prompt": prompt}, timeout=(10, 120))
                if r.status_code == 200:
                    img_b64 = r.json().get("result", {}).get("image")
                    if img_b64:
                        return base64.b64decode(img_b64), None
            except Exception:
                pass
    img_bytes, err = ask_gemini_image(prompt)
    if not err:
        return img_bytes, None
    return None, "Cloudflare's free daily limit is used up for today — try again tomorrow"

# ---- Stitch pictures + voice into a vertical Instagram video ----
def make_reels_video(images, voice_file, out_file="reels.mp4"):
    import os as _os
    import tempfile
    import imageio_ffmpeg
    import moviepy.config as mcfg
    mcfg.FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

    W, H = 1080, 1920
    audio = AudioFileClip(voice_file)
    per = audio.duration / len(images)
    clips = []
    for img_bytes in images:
        tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.write(img_bytes)
        tmp.close()
        clip = ImageClip(tmp.name)
        _os.unlink(tmp.name)
        # zoom the square picture to fill the tall phone screen, keep the middle
        clip = clip.resized((1920, 1920))
        clip = clip.cropped(x_center=960, y_center=960, width=W, height=H)
        clip = clip.with_duration(per)
        clips.append(clip)
    final = concatenate_videoclips(clips, method="chain")
    final = final.with_audio(audio)
    final.write_videofile(out_file, fps=24, codec="libx264",
                          audio_codec="aac", threads=2, preset="ultrafast")
    return out_file

# ---- 1. Browser tab settings ----
st.set_page_config(page_title="My AI Platform", layout="wide")

# ---- 2. The "paint" (CSS) that makes cards look nice ----
st.markdown("""
<style>
    /* Tool cards: real buttons styled as pretty cards (only Home grid, thanks to st-key-card_) */
    div[class*="st-key-card_"] div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #4f46e5, #7c3aed 55%, #a855f7);
        color: #ffffff;
        border-radius: 18px;
        padding: 20px 22px;
        min-height: 170px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        width: 100%;
        white-space: normal;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        transition: transform 0.2s, box-shadow 0.2s, border-color 0.2s;
    }
    div[class*="st-key-card_"] div[data-testid="stButton"] button:hover {
        transform: translateY(-4px) scale(1.02);
        border-color: #f0abfc;
        box-shadow: 0 12px 28px rgba(124, 58, 237, 0.35);
        color: #ffffff;
    }
    div[class*="st-key-card_"] div[data-testid="stButton"] button p {
        width: 100%;
        text-align: left;
        margin: 3px 0;
        font-size: 15px;
        color: #e9d5ff;
    }
    div[class*="st-key-card_"] div[data-testid="stButton"] button p:first-child {
        font-size: 34px;
        margin-bottom: 8px;
    }
    div[class*="st-key-card_"] div[data-testid="stButton"] button strong {
        font-size: 20px;
        color: #ffffff;
    }
    .main-title {
        font-size: 40px;
        font-weight: 800;
        color: #312e81;
    }
    .sub-title {
        color: #6b7280;
        margin-bottom: 30px;
    }
    /* Home hero banner: purple-to-pink gradient */
    .hero {
        background: linear-gradient(135deg, #6d28d9, #9333ea 45%, #ec4899);
        border-radius: 24px;
        padding: 44px 40px 30px;
        margin-bottom: 28px;
        text-align: center;
        color: #ffffff;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.4);
        border-radius: 999px;
        padding: 6px 18px;
        font-size: 13px;
        letter-spacing: 1.5px;
        font-weight: 700;
        margin-bottom: 16px;
    }
    .hero-title {
        font-size: 52px;
        font-weight: 800;
        line-height: 1.1;
    }
    .hero-sub {
        font-size: 17px;
        opacity: 0.95;
        margin-top: 10px;
        line-height: 1.5;
    }
    .hero-stats {
        display: flex;
        justify-content: center;
        gap: 48px;
        margin-top: 26px;
    }
    .stat-num {
        font-size: 26px;
        font-weight: 800;
        display: block;
    }
    .stat-label {
        font-size: 12px;
        opacity: 0.85;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    .pick-title {
        font-size: 22px;
        font-weight: 700;
        color: #312e81;
        margin: 4px 0 14px;
    }
    .tip-strip {
        margin-top: 22px;
        padding: 14px 20px;
        border-radius: 12px;
        background: #f5f3ff;
        color: #4c1d95;
        font-size: 15px;
        border: 1px solid #ddd6fe;
    }
    /* Small gradient banner at the top of each tool page */
    .page-hero {
        background: linear-gradient(135deg, #6d28d9, #9333ea 45%, #ec4899);
        border-radius: 18px;
        padding: 22px 28px;
        margin-bottom: 20px;
        color: #ffffff;
    }
    .ph-title {
        font-size: 30px;
        font-weight: 800;
        line-height: 1.2;
    }
    .ph-sub {
        font-size: 15px;
        opacity: 0.95;
        margin-top: 4px;
    }
    /* The typing box: purple border, rounded, glows when you click it */
    div[data-testid="stChatInput"] {
        border: 2px solid #a78bfa;
        border-radius: 14px;
    }
    div[data-testid="stChatInput"]:focus-within {
        border-color: #7c3aed;
        box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.18);
    }
    div[data-testid="stChatInput"] textarea {
        color: #1f2937;
    }
    /* Sidebar: dark panel matching the theme (every page) */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e1b4b 0%, #312e81 55%, #4c1d95 100%);
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {
        color: #a5b4fc;
    }
    section[data-testid="stSidebar"] label[data-baseweb="radio"] div {
        color: #e0e7ff;
    }
    section[data-testid="stSidebar"] label[data-baseweb="radio"]:hover div {
        color: #ffffff;
    }
    /* Selected menu dot: purple instead of Streamlit's red */
    section[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {
        border-color: #818cf8;
    }
    section[data-testid="stSidebar"] label[data-baseweb="radio"]:has(input:checked) > div:first-child {
        background: #a78bfa;
        border-color: #c4b5fd;
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] button svg {
        fill: #c7d2fe;
    }
    /* Soft lavender wash behind the page */
    div[data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #f5f3ff 0%, #ffffff 380px);
    }
    /* The "Add a photo" folder */
    section[data-testid="stExpander"] details {
        border: 1px solid #ddd6fe;
        border-radius: 12px;
        background: #faf5ff;
    }
    /* "How it works" steps on the Home page */
    .steps {
        display: flex;
        gap: 14px;
        margin-top: 22px;
    }
    .step {
        flex: 1;
        background: #ffffff;
        border: 1px solid #e9d5ff;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 2px 10px rgba(124, 58, 237, 0.07);
    }
    .step-num {
        display: inline-block;
        width: 26px;
        height: 26px;
        line-height: 26px;
        text-align: center;
        border-radius: 50%;
        background: linear-gradient(135deg, #6d28d9, #ec4899);
        color: #ffffff;
        font-weight: 800;
        font-size: 14px;
        margin-bottom: 8px;
    }
    .step-title {
        font-weight: 700;
        color: #312e81;
        font-size: 15px;
    }
    .step-text {
        color: #6b7280;
        font-size: 14px;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---- 3. The left menu ----
menu_items = ["Home", "Chat AI", "Image Studio", "Video Maker", "Resume Maker",
              "Text to Speech", "Thumbnail Maker", "QR Code"]

# "page" remembers where the user is. Cards and the menu both change it.
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

page = st.sidebar.radio("Menu", menu_items, key="page")

# ---- 4. Card click helper (runs BEFORE the page redraws) ----
def go_to(tool_name):
    st.session_state["page"] = tool_name

# ---- 4b. All the tools (data) ----
tools = [
    {"name": "Chat AI",        "emoji": "💬", "desc": "Chat, or write articles & posts", "status": "Ready", "ready": True},
    {"name": "Image Studio",   "emoji": "🎨", "desc": "Make, edit and chat with pictures", "status": "Ready", "ready": True},
    {"name": "Video Maker",    "emoji": "🎬", "desc": "AI video, or your own pics + voice", "status": "Ready", "ready": True},
    {"name": "Resume Maker",   "emoji": "📄", "desc": "Make a job resume in seconds",      "status": "Ready", "ready": True},
    {"name": "Text to Speech", "emoji": "🔊", "desc": "Turn words into a natural voice",   "status": "Ready", "ready": True},
    {"name": "Thumbnail Maker","emoji": "🖼️", "desc": "YouTube thumbnails with big text",  "status": "Ready", "ready": True},
    {"name": "QR Code",        "emoji": "🔗", "desc": "Make a QR code for any link",       "status": "Ready", "ready": True},
]

# ---- 5. Show the page ----
if page == "Home":
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">⚡ 7 FREE AI TOOLS — ONE PLATFORM</div>
        <div class="hero-title">AI Platform</div>
        <div class="hero-sub">Make pictures, videos, articles, resumes and more —<br>just type in English. No login, no payment.</div>
        <div class="hero-stats">
            <div class="stat"><span class="stat-num">7</span><span class="stat-label">AI Tools</span></div>
            <div class="stat"><span class="stat-num">100%</span><span class="stat-label">Free</span></div>
            <div class="stat"><span class="stat-num">0</span><span class="stat-label">Sign-ups</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="pick-title">Pick a tool to start 👇</div>', unsafe_allow_html=True)

    # Grid of 3 cards per row - cards are real buttons that switch pages safely
    cols = st.columns(3)
    for i, tool in enumerate(tools):
        with cols[i % 3]:
            label = f"{tool['emoji']}\n\n**{tool['name']}**  \n{tool['desc']}  \n🟢 {tool['status']}"
            st.button(label, key=f"card_{tool['name']}",
                      on_click=go_to, args=(tool["name"],), use_container_width=True)

    st.markdown('<div class="tip-strip">💡 Tip: everything here is free — start with '
                '<b>Chat AI</b> and just type a question.</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="steps">
        <div class="step">
            <div class="step-num">1</div>
            <div class="step-title">Pick a tool</div>
            <div class="step-text">Click a card above, or use the menu on the left.</div>
        </div>
        <div class="step">
            <div class="step-num">2</div>
            <div class="step-title">Type in English</div>
            <div class="step-text">Say what you want — the AI figures out the rest.</div>
        </div>
        <div class="step">
            <div class="step-num">3</div>
            <div class="step-title">Download it</div>
            <div class="step-text">Save your picture, video, text or resume in one click.</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

elif page == "Chat AI":
    st.markdown('<div class="page-hero"><div class="ph-title">💬 Chat AI</div>'
                '<div class="ph-sub">Talk with AI like ChatGPT, or switch to Write mode '
                'for articles, posts and emails.</div></div>',
                unsafe_allow_html=True)

    chat_mode = st.radio("What do you want to do?",
                         ["💬 Chat", "✍️ Write (articles, posts, emails)"],
                         horizontal=True, key="chat_mode")

    if chat_mode.startswith("✍️"):
        # ---- Write mode (the old Text & Articles tool, folded in here) ----
        kind = st.selectbox("What do you want to write?",
                            ["Blog Article", "News Story", "Social Media Post",
                             "YouTube Title + Description", "Email"])
        topic = st.text_input("About what? (example: healthy food, cricket, mobile phones)",
                              key="write_topic")
        length = st.select_slider("How long?", options=["Short", "Medium", "Long"])

        if st.button("Write it!", key="write_btn"):
            if not topic.strip():
                st.warning("Type a topic first!")
            else:
                with st.spinner("AI is writing... please wait"):
                    instructions = (f"You are a professional writer. "
                                    f"Write a {kind} about: {topic}. "
                                    f"Length: {length}. "
                                    f"Make it interesting and easy to read. Use headings and short paragraphs.")
                    result = ask_text_ai(instructions)
                    if not result:
                        st.error(BUSY_MSG)
                    else:
                        st.markdown(result)
                        st.download_button("Download as .txt", result.encode("utf-8"),
                                           file_name="my_text.txt", mime="text/plain")
    else:
        # ---- Chat mode ----
        # Optional: upload a picture so the AI can see it.
        # Tucked inside a folder so the typing box always fits on the screen.
        with st.expander("📷 Add a photo to your message (optional)"):
            chat_img = st.file_uploader("Upload a photo (JPG or PNG)",
                                        type=["jpg", "jpeg", "png"], key="chat_img")

        # Remember old messages so they don't disappear
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Show all old messages
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                if m.get("image"):
                    st.image(m["image"], width=220)
                st.write(m["content"])

        # The typing box at the bottom
        prompt = st.chat_input("Type your message here...")

        if prompt:
            # Show what the user typed (and the picture, if any)
            msg = {"role": "user", "content": prompt}
            if chat_img is not None:
                msg["image"] = chat_img.getvalue()
            st.session_state.messages.append(msg)
            with st.chat_message("user"):
                if msg.get("image"):
                    st.image(msg["image"], width=220)
                st.write(prompt)

            # Get the AI answer (with eyes if a picture was added)
            if msg.get("image"):
                reply = ask_gemini_vision(prompt, msg["image"])
            else:
                reply = ask_text_ai(prompt) or BUSY_MSG

            st.session_state.messages.append({"role": "assistant", "content": reply})
            with st.chat_message("assistant"):
                st.write(reply)

elif page == "Image Studio":
    st.markdown('<div class="main-title">Image Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Make a new picture, or upload your own photo and ask AI to change it.</div>',
                unsafe_allow_html=True)

    tab_new, tab_edit, tab_chat = st.tabs(["Make a new picture", "Edit my photo", "Chat with pictures"])

    with tab_new:
        prompt = st.text_input("What picture do you want? (English works best)", key="ig_prompt")
        if st.button("Generate Image", key="ig_gen_btn"):
            if not prompt.strip():
                st.warning("Type something first!")
            else:
                with st.spinner("AI is drawing... please wait"):
                    # Try each free provider silently, in order, until one works
                    daily_limit = False
                    img_bytes, err = ask_gemini_image(prompt)
                    if err:
                        img_bytes, err = ask_cf_image(prompt)
                        if err and "10,000 neurons" in err:
                            daily_limit = True
                    if err:
                        img_bytes, err = ask_together_image(prompt)
                    if err:
                        url = f"https://image.pollinations.ai/prompt/{quote(prompt)}?width=768&height=768&nologo=true"
                        try:
                            r = requests.get(url, timeout=180)
                            if r.status_code == 200:
                                img_bytes, err = r.content, None
                        except Exception:
                            pass
                    if err:
                        if daily_limit:
                            st.error("Cloudflare's free daily limit is used up for today. "
                                     "It resets every day — please try again tomorrow!")
                        else:
                            st.error("Sorry, the free image services are busy right now. Please try again in a minute.")
                    else:
                        st.image(img_bytes, caption=prompt, use_container_width=True)
                        st.download_button("Download Image", img_bytes,
                                           file_name="my_ai_image.png", mime="image/png")

    with tab_edit:
        my_photo = st.file_uploader("Upload your photo (JPG or PNG)",
                                    type=["jpg", "jpeg", "png"], key="ig_photo")
        if my_photo is not None:
            st.image(my_photo, caption="Your photo", use_container_width=True)
            change = st.text_input("What change do you want? (example: change the sky to sunset, make me look younger)",
                                   key="ig_change")
            if st.button("Edit My Photo", key="ig_edit_btn", type="primary"):
                if not change.strip():
                    st.warning("Describe the change you want first!")
                else:
                    with st.spinner("AI is editing your photo... please wait (up to 2 minutes)"):
                        edited, err = ask_edit_photo(change, my_photo.getvalue())
                    if err == "daily limit":
                        st.error("Cloudflare's free daily limit is used up for today. "
                                 "It resets every day — please try again tomorrow! "
                                 "Meanwhile, use 'Write text on your photo' below — it never runs out.")
                    elif err:
                        st.error("Sorry, the photo editor is busy right now. Please try again in a minute.")
                    else:
                        st.image(edited, caption="Your edited photo", use_container_width=True)
                        st.download_button("Download Edited Photo", edited,
                                           file_name="my_edited_photo.jpg", mime="image/jpeg")

            st.markdown("### Write text on your photo (free, always works)")
            st.caption("No AI needed for this — your words are drawn straight onto the photo, "
                       "like a name or a title. Works instantly, no daily limit.")
            text_words = st.text_input("What words do you want on the photo?",
                                       key="ig_text_words",
                                       placeholder="example: Farooq Khan")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                text_pos = st.selectbox("Where?", ["Bottom", "Middle", "Top"], key="ig_text_pos")
            with col_b:
                text_color = st.selectbox("Color", ["White", "Black", "Red", "Yellow", "Blue"],
                                          key="ig_text_color")
            with col_c:
                text_size = st.selectbox("Size", ["Small", "Medium", "Big"], key="ig_text_size")
            if st.button("Write text on photo", key="ig_text_btn", type="primary"):
                if not text_words.strip():
                    st.warning("Type the words first!")
                else:
                    out_png = write_text_on_photo(my_photo.getvalue(), text_words.strip(),
                                                  position=text_pos.lower(),
                                                  color=text_color.lower(),
                                                  size=text_size)
                    st.success("Done! Your text is on the photo — see it and download it below.")
                    st.image(out_png, caption="Your photo with text", use_container_width=True)
                    st.download_button("Download photo with text", out_png,
                                       file_name="photo_with_text.png", mime="image/png")

    with tab_chat:
        st.caption("Like ChatGPT: make a picture, then keep telling the AI what to change "
                   "on the SAME picture. It remembers the picture for you.")

        if st.session_state.get("ig_chat_img") is None:
            first = st.text_input("Describe your first picture (English works best)",
                                  key="ig_chat_first",
                                  placeholder="example: a cup of coffee on a wooden table")
            if st.button("Make it", key="ig_chat_make_btn", type="primary"):
                if not first.strip():
                    st.warning("Describe the picture first!")
                else:
                    with st.spinner("AI is drawing your picture... please wait"):
                        daily_limit = False
                        img_bytes, err = ask_gemini_image(first)
                        if err:
                            img_bytes, err = ask_cf_image(first)
                            if err and "10,000 neurons" in err:
                                daily_limit = True
                        if err:
                            url = (f"https://image.pollinations.ai/prompt/{quote(first)}"
                                   f"?width=768&height=768&nologo=true")
                            try:
                                r = requests.get(url, timeout=180)
                                if r.status_code == 200:
                                    img_bytes, err = r.content, None
                            except Exception:
                                pass
                        if err:
                            if daily_limit:
                                st.error("Cloudflare's free daily limit is used up for today. "
                                         "It resets every day — please try again tomorrow!")
                            else:
                                st.error("Sorry, the free image services are busy right now. "
                                         "Please try again in a minute.")
                        else:
                            st.session_state["ig_chat_img"] = img_bytes
                            st.success("Picture made! Now tell the AI what to change on it.")
        else:
            st.image(st.session_state["ig_chat_img"], caption="Your picture",
                     use_container_width=True)
            inst = st.text_input("What should the AI change? (example: make the sky a sunset, "
                                 "put a cat next to it)",
                                 key="ig_chat_inst")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Change it", key="ig_chat_change_btn", type="primary"):
                    if not inst.strip():
                        st.warning("Tell the AI what to change first!")
                    else:
                        with st.spinner("AI is changing your picture... please wait (up to 2 minutes)"):
                            new_img, err = ask_edit_photo(inst, st.session_state["ig_chat_img"])
                        if err == "daily limit":
                            st.error("Cloudflare's free daily limit is used up for today. "
                                     "It resets every day — please try again tomorrow!")
                        elif err:
                            st.error("Sorry, the photo editor is busy right now. "
                                     "Please try again in a minute.")
                        else:
                            st.session_state["ig_chat_img"] = new_img
                            st.success("Changed! Tell the AI what to change next.")
            with col2:
                if st.button("Start over", key="ig_chat_reset_btn"):
                    st.session_state["ig_chat_img"] = None
                    st.rerun()
            st.download_button("Download this picture", st.session_state["ig_chat_img"],
                               file_name="my_ai_picture.jpg", mime="image/jpeg")

elif page == "Resume Maker":
    st.markdown('<div class="page-hero"><div class="ph-title">📄 Resume Maker</div>'
                '<div class="ph-sub">Fill in the boxes and AI will write your resume. '
                'After that, just type what you want changed.</div></div>',
                unsafe_allow_html=True)

    # The resume lives here so it stays on screen while you ask for changes
    if "resume_text" not in st.session_state:
        st.session_state["resume_text"] = ""
    if "resume_log" not in st.session_state:
        st.session_state["resume_log"] = []

    # st.form = a box that waits. Nothing happens until you press the button.
    with st.form("resume_form"):
        st.markdown("**About you**")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your full name")
            job_title = st.text_input("What job do you want? (example: Web Developer)")
        with col2:
            email = st.text_input("Email ID")
            phone = st.text_input("Phone number")
        address = st.text_input("Address",
                                placeholder="example: House 12, MG Road, Mumbai")

        st.markdown("**Education**")
        education = st.text_area("Education",
                                 placeholder="example: B.A. English, City College, 2020")

        st.markdown("**Skills**")
        skills = st.text_area("Skills (separate with commas)",
                              placeholder="example: Python, Microsoft Excel, English, teamwork")

        st.markdown("**Work experience**")
        experience = st.text_area("Work experience",
                                  placeholder="example: Cashier at FreshMart, 2022-2024")

        st.markdown("**Personal details**")
        col3, col4 = st.columns(2)
        with col3:
            father_name = st.text_input("Father's name")
            dob = st.text_input("Date of birth", placeholder="example: 15 August 2002")
        with col4:
            gender = st.selectbox("Gender", ["Male", "Female"])
            marital = st.selectbox("Marital status", ["Single", "Married"])
        strong_points = st.text_area("Strong points",
                                     placeholder="example: hardworking, quick learner, honest")

        submitted = st.form_submit_button("Write my resume", type="primary")

    # The chat box: people type what they want built into their resume
    change = st.chat_input("Want a change? Type it here (example: add a Projects section)")

    if submitted:
        if not name.strip():
            st.warning("Type your name first!")
        else:
            with st.spinner("AI is writing your resume... please wait"):
                # f-string = put a variable inside text with { }
                # The "prompt trick": tell the AI its job + give it rules, and it writes much better
                instructions = (
                    f"You are a professional resume writer. Write a clean, professional "
                    f"resume for this person.\n\n"
                    f"Name: {name}\n"
                    f"Job they want: {job_title}\n"
                    f"Email: {email}\n"
                    f"Phone: {phone}\n"
                    f"Address: {address}\n"
                    f"Education: {education}\n"
                    f"Skills: {skills}\n"
                    f"Experience: {experience}\n"
                    f"Father's name: {father_name}\n"
                    f"Date of birth: {dob}\n"
                    f"Gender: {gender}\n"
                    f"Marital status: {marital}\n"
                    f"Strong points: {strong_points}\n\n"
                    f"Rules:\n"
                    f"1. Fix all spelling and grammar mistakes, but keep the facts the same.\n"
                    f"2. Use '## Section Name' for every heading and '-' for every bullet point.\n"
                    f"3. Make it impressive and professional.\n"
                    f"4. Keep it under one page if possible.\n"
                    f"5. Write the resume text only, no extra comments.\n"
                    f"6. NEVER invent company names, dates, numbers or facts. If a detail is "
                    f"missing, leave that line or section OUT completely - never write "
                    f"[brackets], placeholders like [Year], or notes to the reader.\n"
                    f"7. End the resume with a '## Personal Details' section containing "
                    f"father's name, date of birth, gender, marital status and strong points."
                )
                draft = ask_text_ai(instructions)
                if not draft:
                    st.error(BUSY_MSG)
                else:
                    st.session_state["resume_text"] = draft
                    st.session_state["resume_log"] = []
                    st.session_state["resume_header"] = {
                        "name": name.strip(), "title": job_title.strip(),
                        "email": email.strip(), "phone": phone.strip(),
                        "contact": "   |   ".join(p.strip() for p in (email, phone, address) if p and p.strip()),
                    }

    if change:
        if not st.session_state["resume_text"]:
            st.warning("Fill the form above and write your first resume, then ask for changes.")
        else:
            with st.spinner("AI is fixing your resume... please wait"):
                ask = (
                    f"Here is my current resume:\n\n{st.session_state['resume_text']}\n\n"
                    f"My request: {change}\n\n"
                    f"Rules:\n"
                    f"1. Do the change I asked for. Keep the other facts as they are.\n"
                    f"2. Keep the same format: '## Section Name' headings and '-' bullets.\n"
                    f"3. Fix spelling and grammar.\n"
                    f"4. NEVER invent facts. If a detail is missing, leave it out - "
                    f"never write [brackets] or notes to the reader.\n"
                    f"5. Return the full resume text only, no comments."
                )
                updated = ask_text_ai(ask)
            if not updated:
                st.error(BUSY_MSG)
            else:
                st.session_state["resume_text"] = updated
                st.session_state["resume_log"].append(change)

    if st.session_state["resume_text"]:
        st.markdown("### Your resume")
        st.markdown(st.session_state["resume_text"])

        # Download as PDF (for job applications) or TXT
        header = st.session_state.get("resume_header") or {}
        pdf_bytes = make_resume_pdf(st.session_state["resume_text"], header)
        safe_name = (header.get("name") or "My_Resume").replace(" ", "_")
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button("Download as PDF", pdf_bytes, key="resume_pdf_btn",
                               file_name=f"Resume_{safe_name}.pdf",
                               mime="application/pdf")
        with col_b:
            st.download_button("Download as .txt",
                               st.session_state["resume_text"].encode("utf-8"),
                               key="resume_txt_btn",
                               file_name=f"Resume_{safe_name}.txt",
                               mime="text/plain")
        if st.session_state["resume_log"]:
            st.caption("Changes you asked for:  " + "  •  ".join(st.session_state["resume_log"]))
    else:
        st.info("👆 Fill in your details above and press **Write my resume**. "
                "After that you can type any change you want in the box at the bottom.")

elif page == "Video Maker":
    st.markdown('<div class="main-title">Video Maker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Make a short vertical video for Instagram Reels or '
                'YouTube Shorts — let AI do everything, or use your own script and pictures.</div>',
                unsafe_allow_html=True)

    platform = st.radio("Where will you post it?",
                        ["Instagram Reels", "YouTube Shorts"],
                        horizontal=True, key="vm_platform")
    line_count = 5 if platform.startswith("Instagram") else 6

    mode = st.radio("How do you want to make it?",
                    ["Easy: AI does everything", "My way: my own script + pictures"],
                    key="vm_mode")

    if mode.startswith("Easy"):
        topic = st.text_input("What is your video about? (example: 5 amazing facts about space)",
                              key="vm_topic")

        if st.button("Write my script", key="vm_script_btn"):
            if not topic.strip():
                st.warning("Type a topic first!")
            else:
                with st.spinner("AI is writing the script... please wait"):
                    instructions = (
                        f"You are a viral {platform} script writer. "
                        f"Write a script for a short vertical video about: {topic}\n\n"
                        f"Rules:\n"
                        f"1. Exactly {line_count} lines. Each line is one scene "
                        f"(later, one picture will show per line).\n"
                        f"2. Make line 1 a strong hook that makes people stop scrolling.\n"
                        f"3. Keep every line under 12 words. Simple, punchy English.\n"
                        f"4. Write ONLY the {line_count} lines, each starting with '- '. "
                        f"No headings, no numbers, no extra text."
                    )
                    st.session_state["vm_script"] = ask_text_ai(instructions) or BUSY_MSG
    else:
        st.markdown("**Step 1:** Write your script. One line = one scene of the video.")
        my_script = st.text_area("Your script (one line per scene, example:  Welcome to my shop!)",
                                 height=180, key="vm_my_script")
        st.markdown("**Step 2:** Upload your pictures in the same order as your lines.")
        my_files = st.file_uploader("Upload pictures (JPG or PNG)",
                                    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                    key="vm_my_files")
        st.caption("Tip: upload one picture per line. The voice reads the lines while your pictures show.")

    # Get the script lines from whatever source the user chose
    if mode.startswith("Easy"):
        lines = parse_script_lines(st.session_state["vm_script"]) if "vm_script" in st.session_state else None
    else:
        lines = parse_script_lines(my_script) if my_script.strip() else None

    if lines:
        st.markdown("### Your script")
        st.markdown("\n".join("- " + ln for ln in lines))

        st.markdown("### Voice")
        voice_label = st.selectbox("Choose a voice", [
            "en-US-JennyNeural (US woman)",
            "en-US-GuyNeural (US man)",
            "en-IN-NeerjaNeural (India woman)",
            "en-IN-PrabhatNeural (India man)"], key="vm_voice_sel")
        voice_code = voice_label.split(" ")[0]

        if st.button("Make voice", key="vm_voice_btn"):
            with st.spinner("AI is speaking... please wait"):
                spoken_text = ". ".join(lines)
                voice_file = make_voice(spoken_text, voice=voice_code)
                st.session_state["vm_voice"] = voice_file
            st.success("Voice ready! Play it below.")
            st.audio(open(voice_file, "rb").read(), format="audio/mp3")

        st.markdown("### Pictures")
        if mode.startswith("Easy"):
            if st.button("Make pictures", key="vm_pics_btn"):
                with st.spinner("AI is drawing the scenes... this can take 1-3 minutes"):
                    progress = st.progress(0)
                    images = []
                    for i, line in enumerate(lines):
                        prompt = (f"{platform} scene, cinematic photo style: {line}. "
                                  f"Vibrant colors, dramatic lighting, NO text on the image")
                        img_bytes, err = ask_scene_image(prompt)
                        if err:
                            st.error(f"Scene {i+1} failed ({err}). Try again in a minute.")
                            break
                        images.append(img_bytes)
                        progress.progress((i + 1) / len(lines))
                    progress.empty()
                    if images:
                        st.session_state["vm_images"] = images

        # Pictures from whatever source the user chose
        if mode.startswith("Easy"):
            images = st.session_state["vm_images"] if "vm_images" in st.session_state else None
        else:
            images = [f.getvalue() for f in my_files] if my_files else None

        if images:
            st.success(f"{len(images)} scene pictures ready!")
            cols = st.columns(5)
            for i, img in enumerate(images):
                with cols[i % 5]:
                    st.image(img, use_container_width=True)

        st.markdown("### Video")
        if st.button("Make video", key="vm_video_btn", type="primary"):
            if not images:
                st.warning("Make pictures first!" if mode.startswith("Easy")
                           else "Upload at least one picture first!")
            else:
                with st.spinner("Putting voice + pictures together... this takes 1-2 minutes"):
                    # No voice yet? Make one quietly with the chosen voice.
                    if "vm_voice" not in st.session_state:
                        spoken_text = ". ".join(lines)
                        st.session_state["vm_voice"] = make_voice(spoken_text, voice=voice_code)
                    out_name = "reels.mp4" if platform.startswith("Instagram") else "shorts.mp4"
                    video_file = make_reels_video(images, st.session_state["vm_voice"],
                                                  out_file=out_name)
                    st.session_state["vm_video"] = video_file
                st.success("Your video is ready!")
                st.video(open(video_file, "rb").read(), format="video/mp4")
                st.download_button("Download video (.mp4)",
                                   open(video_file, "rb").read(),
                                   file_name=f"my_{out_name}", mime="video/mp4")

elif page == "Text to Speech":
    st.markdown('<div class="main-title">Text to Speech</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Type anything and get a natural-sounding voice as an '
                'MP3. Free, and it never runs out of credits.</div>', unsafe_allow_html=True)

    tts_text = st.text_area("What should the voice say?", height=180, key="tts_text",
                            placeholder="Example: Hello everyone, welcome to my channel!")
    tts_label = st.selectbox("Choose a voice", [
        "en-US-JennyNeural (US woman)",
        "en-US-GuyNeural (US man)",
        "en-IN-NeerjaNeural (India woman)",
        "en-IN-PrabhatNeural (India man)"], key="tts_voice_sel")

    if st.button("Make audio", key="tts_btn", type="primary"):
        if not tts_text.strip():
            st.warning("Type some words first!")
        else:
            with st.spinner("AI is speaking... please wait"):
                vf = make_voice(tts_text.strip(), voice=tts_label.split(" ")[0],
                                filename="speech.mp3")
            st.success("Your audio is ready!")
            st.audio(open(vf, "rb").read(), format="audio/mp3")
            st.download_button("Download audio (.mp3)", open(vf, "rb").read(),
                               file_name="my_speech.mp3", mime="audio/mpeg")

elif page == "Thumbnail Maker":
    st.markdown('<div class="main-title">Thumbnail Maker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Make a 1280x720 YouTube thumbnail: an AI background '
                'with your big title text on top.</div>', unsafe_allow_html=True)

    th_title = st.text_input("Big title text on the thumbnail", key="th_title",
                             placeholder="Example: I TRIED THIS FOR 30 DAYS!")
    th_bg = st.text_input("Background idea (the AI picture behind the text)", key="th_bg",
                          placeholder="Example: surprised face, bright studio, dramatic lighting")
    col_a, col_b = st.columns(2)
    with col_a:
        th_color = st.selectbox("Text color", ["White", "Black", "Yellow", "Red"], key="th_color")
    with col_b:
        th_pos = st.selectbox("Text position", ["Middle", "Top", "Bottom"], key="th_pos")

    if st.button("Make thumbnail", key="th_btn", type="primary"):
        if not th_title.strip():
            st.warning("Type the title text first!")
        else:
            with st.spinner("Making your background picture... please wait"):
                bg_prompt = th_bg.strip() or th_title.strip()
                scene = (f"YouTube thumbnail background, bold, colorful, high contrast, "
                         f"16:9, NO text: {bg_prompt}")
                img_bytes, err = ask_scene_image(scene)
            if err:
                st.error("Sorry, the picture service is busy or out of daily free credits. "
                         "It resets every day — please try again tomorrow.")
            else:
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((1280, 720))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                out = write_text_on_photo(buf.getvalue(), th_title.strip(),
                                          position=th_pos.lower(), color=th_color.lower(),
                                          size="Big")
                st.success("Your thumbnail is ready!")
                st.image(out, use_container_width=True)
                st.download_button("Download thumbnail (.png)", out,
                                   file_name="my_thumbnail.png", mime="image/png")

elif page == "QR Code":
    st.markdown('<div class="main-title">QR Code</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Turn any link or text into a QR code you can download '
                'and print. Free and instant.</div>', unsafe_allow_html=True)

    qr_data = st.text_input("Link or text", key="qr_data",
                            placeholder="Example: https://yoursite.com")
    if st.button("Make QR code", key="qr_btn", type="primary"):
        if not qr_data.strip():
            st.warning("Type a link or some text first!")
        else:
            try:
                import io
                import qrcode
                qr = qrcode.QRCode(box_size=10, border=2)
                qr.add_data(qr_data.strip())
                qr.make(fit=True)
                img = qr.make_image(fill_color="black", back_color="white")
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                st.success("Here is your QR code — scan it or download it.")
                st.image(buf.getvalue())
                st.download_button("Download QR (.png)", buf.getvalue(),
                                   file_name="qr_code.png", mime="image/png")
            except ImportError:
                st.error("The QR library is not installed yet. Add 'qrcode' to requirements.txt.")

else:
    # Placeholder page for every tool (we build them one by one)
    st.markdown(f'<div class="main-title">{page}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">This tool is not built yet. It comes in the next steps.</div>',
                unsafe_allow_html=True)
