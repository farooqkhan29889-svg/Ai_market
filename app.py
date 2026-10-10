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

# ---- Cut a part out of an uploaded video ----
def cut_video(input_file, start, end, out_file="edited.mp4"):
    import imageio_ffmpeg
    import moviepy.config as mcfg
    mcfg.FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_file)
    cut = clip.subclipped(start, end)
    cut.write_videofile(out_file, codec="libx264", audio_codec="aac",
                        preset="ultrafast", threads=2)
    clip.close()
    return out_file

# ---- How long is a video? (needed to set the trim slider) ----
def get_video_duration(input_file):
    import imageio_ffmpeg
    import moviepy.config as mcfg
    mcfg.FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_file)
    d = clip.duration
    clip.close()
    return d

# ---- Put an AI voice over a video ----
# If the voice is longer than the video, the last frame freezes until it finishes.
def add_voice_to_video(video_file, voice_file, out_file="edited_with_voice.mp4",
                       keep_original=False):
    import imageio_ffmpeg
    import moviepy.config as mcfg
    mcfg.FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
    from moviepy import (VideoFileClip, AudioFileClip, CompositeAudioClip,
                         concatenate_videoclips)
    video = VideoFileClip(video_file)
    voice = AudioFileClip(voice_file)
    if voice.duration > video.duration:
        frozen = video.to_ImageClip(t=video.duration - 0.1)
        frozen = frozen.with_duration(voice.duration - video.duration)
        video = concatenate_videoclips([video, frozen])
    if keep_original and video.audio is not None:
        audio = CompositeAudioClip([voice, video.audio.volumex(0.25)])
        audio = audio.with_duration(video.duration)
    else:
        audio = voice
    final = video.with_audio(audio)
    final.write_videofile(out_file, codec="libx264", audio_codec="aac",
                          preset="ultrafast", threads=2)
    final.close()
    return out_file

# ---- Turn a video vertical (9:16) for Reels and Shorts ----
def make_vertical(input_file, out_file="vertical.mp4"):
    import imageio_ffmpeg
    import moviepy.config as mcfg
    mcfg.FFMPEG_BINARY = imageio_ffmpeg.get_ffmpeg_exe()
    from moviepy import VideoFileClip
    clip = VideoFileClip(input_file)
    # Cut the middle 9:16 piece of the picture, then stretch it to 1080x1920
    target = 9 / 16
    if clip.w / clip.h > target:
        cw = int(clip.h * target)
        ch = clip.h
    else:
        cw = clip.w
        ch = int(clip.w / target)
    cropped = clip.cropped(width=cw, height=ch,
                           x_center=clip.w / 2, y_center=clip.h / 2)
    final = cropped.resized((1080, 1920))
    final.write_videofile(out_file, codec="libx264", audio_codec="aac",
                          preset="ultrafast", threads=2)
    clip.close()
    return out_file

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
menu_items = ["Home", "Chat AI", "Image Generator", "Text & Articles",
              "Resume Maker", "Reels Maker", "YouTube Shorts", "Video Editing"]

# "page" remembers where the user is. Cards and the menu both change it.
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

page = st.sidebar.radio("Menu", menu_items, key="page")

# ---- 4. Card click helper (runs BEFORE the page redraws) ----
def go_to(tool_name):
    st.session_state["page"] = tool_name

# ---- 4b. All the tools (data) ----
tools = [
    {"name": "Chat AI",       "emoji": "💬", "desc": "Talk with AI like ChatGPT",          "status": "Ready",  "ready": True},
    {"name": "Image Generator","emoji": "🎨", "desc": "Type words, get a picture",         "status": "Ready",  "ready": True},
    {"name": "Text & Articles","emoji": "✍️", "desc": "Write articles, news, posts",       "status": "Ready",  "ready": True},
    {"name": "Resume Maker",  "emoji": "📄", "desc": "Make a job resume in seconds",       "status": "Ready", "ready": True},
    {"name": "Reels Maker",   "emoji": "🎬", "desc": "AI video or your own pics",           "status": "Ready",   "ready": True},
    {"name": "YouTube Shorts","emoji": "📱", "desc": "AI video or your own pics",           "status": "Ready",   "ready": True},
    {"name": "Video Editing", "emoji": "✂️", "desc": "Cut, voice-over, vertical videos",     "status": "Ready", "ready": True},
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
                '<div class="ph-sub">Talk with AI. Add a photo to ask about it, like ChatGPT.</div></div>',
                unsafe_allow_html=True)

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

elif page == "Image Generator":
    st.markdown('<div class="main-title">Image Generator</div>', unsafe_allow_html=True)
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

elif page == "Text & Articles":
    st.markdown('<div class="main-title">Text & Articles</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Write articles, news and posts in seconds.</div>',
                unsafe_allow_html=True)

    kind = st.selectbox("What do you want to write?",
                        ["Blog Article", "News Story", "Social Media Post",
                         "YouTube Title + Description", "Email"])
    topic = st.text_input("About what? (example: healthy food, cricket, mobile phones)")
    length = st.select_slider("How long?", options=["Short", "Medium", "Long"])

    if st.button("Write it!"):
        if not topic.strip():
            st.warning("Type a topic first!")
        else:
            with st.spinner("AI is writing... please wait"):
                # The "prompt trick": tell the AI its job clearly and it writes much better
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

elif page == "Reels Maker":
    st.markdown('<div class="main-title">Reels Maker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Two ways to make a Reels: let AI do everything, '
                'or write your own script and use your own pictures.</div>',
                unsafe_allow_html=True)

    mode = st.radio("How do you want to make it?",
                    ["Easy: AI does everything", "My way: my own script + pictures"],
                    key="reels_mode")

    if mode.startswith("Easy"):
        topic = st.text_input("What is your reels about? (example: 5 amazing facts about space)",
                              key="reels_topic")

        if st.button("Write my script"):
            if not topic.strip():
                st.warning("Type a topic first!")
            else:
                with st.spinner("AI is writing the script... please wait"):
                    instructions = (
                        f"You are a viral Instagram Reels script writer. "
                        f"Write a script for a short video about: {topic}\n\n"
                        f"Rules:\n"
                        f"1. Exactly 5 lines. Each line is one scene "
                        f"(later, one picture will show per line).\n"
                        f"2. Make line 1 a strong hook that makes people stop scrolling.\n"
                        f"3. Keep every line under 12 words. Simple, punchy English.\n"
                        f"4. Write ONLY the 5 lines, each starting with '- '. "
                        f"No headings, no numbers, no extra text."
                    )
                    st.session_state["reels_script"] = ask_text_ai(instructions) or BUSY_MSG
    else:
        st.markdown("**Step 1:** Write your script. One line = one scene of the video.")
        my_script = st.text_area("Your script (one line per scene, example:  Welcome to my shop!)",
                                 height=180, key="reels_my_script")
        st.markdown("**Step 2:** Upload your pictures in the same order as your lines.")
        my_files = st.file_uploader("Upload pictures (JPG or PNG)",
                                    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                    key="reels_my_files")
        st.caption("Tip: upload one picture per line. The voice reads the lines while your pictures show.")

    # Get the script lines from whatever source the user chose
    if mode.startswith("Easy"):
        lines = parse_script_lines(st.session_state["reels_script"]) if "reels_script" in st.session_state else None
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
            "en-IN-PrabhatNeural (India man)"], key="reels_voice_sel")
        voice_code = voice_label.split(" ")[0]

        if st.button("Make voice", key="reels_voice_btn"):
            with st.spinner("AI is speaking... please wait"):
                spoken_text = ". ".join(lines)
                voice_file = make_voice(spoken_text, voice=voice_code)
                st.session_state["reels_voice"] = voice_file
            st.success("Voice ready! Play it below.")
            st.audio(open(voice_file, "rb").read(), format="audio/mp3")

        st.markdown("### Pictures")
        if mode.startswith("Easy"):
            if st.button("Make pictures", key="reels_pics_btn"):
                with st.spinner("AI is drawing the scenes... this can take 1-3 minutes"):
                    progress = st.progress(0)
                    images = []
                    for i, line in enumerate(lines):
                        prompt = (f"Instagram Reels scene, cinematic photo style: {line}. "
                                  f"Vibrant colors, dramatic lighting, NO text on the image")
                        img_bytes, err = ask_scene_image(prompt)
                        if err:
                            st.error(f"Scene {i+1} failed ({err}). Try again in a minute.")
                            break
                        images.append(img_bytes)
                        progress.progress((i + 1) / len(lines))
                    progress.empty()
                    if images:
                        st.session_state["reels_images"] = images

        # Pictures from whatever source the user chose
        if mode.startswith("Easy"):
            images = st.session_state["reels_images"] if "reels_images" in st.session_state else None
        else:
            images = [f.getvalue() for f in my_files] if my_files else None

        if images:
            st.success(f"{len(images)} scene pictures ready!")
            cols = st.columns(5)
            for i, img in enumerate(images):
                with cols[i % 5]:
                    st.image(img, use_container_width=True)

        st.markdown("### Video")
        if st.button("Make video", key="reels_video_btn", type="primary"):
            if not images:
                st.warning("Make pictures first!" if mode.startswith("Easy")
                           else "Upload at least one picture first!")
            else:
                with st.spinner("Putting voice + pictures together... this takes 1-2 minutes"):
                    # No voice yet? Make one quietly with the chosen voice.
                    if "reels_voice" not in st.session_state:
                        spoken_text = ". ".join(lines)
                        st.session_state["reels_voice"] = make_voice(spoken_text, voice=voice_code)
                    video_file = make_reels_video(images, st.session_state["reels_voice"])
                    st.session_state["reels_video"] = video_file
                st.success("Your Reels video is ready!")
                st.video(open(video_file, "rb").read(), format="video/mp4")
                st.download_button("Download Reels (.mp4)",
                                   open(video_file, "rb").read(),
                                   file_name="my_reels.mp4", mime="video/mp4")

elif page == "YouTube Shorts":
    st.markdown('<div class="main-title">YouTube Shorts</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Two ways to make a Shorts: let AI do everything, '
                'or write your own script and use your own pictures.</div>',
                unsafe_allow_html=True)

    mode = st.radio("How do you want to make it?",
                    ["Easy: AI does everything", "My way: my own script + pictures"],
                    key="yt_mode")

    if mode.startswith("Easy"):
        topic = st.text_input("What is your Shorts about? (example: 3 tips to study better)",
                              key="yt_topic")

        if st.button("Write my script", key="yt_script_btn"):
            if not topic.strip():
                st.warning("Type a topic first!")
            else:
                with st.spinner("AI is writing the script... please wait"):
                    instructions = (
                        f"You are a viral YouTube Shorts script writer. "
                        f"Write a script for a short vertical video about: {topic}\n\n"
                        f"Rules:\n"
                        f"1. Exactly 6 lines. Each line is one scene "
                        f"(later, one picture will show per line).\n"
                        f"2. Make line 1 a strong hook that makes people keep watching.\n"
                        f"3. Keep every line under 12 words. Simple, punchy English.\n"
                        f"4. Write ONLY the 6 lines, each starting with '- '. "
                        f"No headings, no numbers, no extra text."
                    )
                    st.session_state["yt_script"] = ask_text_ai(instructions) or BUSY_MSG
    else:
        st.markdown("**Step 1:** Write your script. One line = one scene of the video.")
        my_script = st.text_area("Your script (one line per scene, example:  Welcome to my channel!)",
                                 height=180, key="yt_my_script")
        st.markdown("**Step 2:** Upload your pictures in the same order as your lines.")
        my_files = st.file_uploader("Upload pictures (JPG or PNG)",
                                    type=["jpg", "jpeg", "png"], accept_multiple_files=True,
                                    key="yt_my_files")
        st.caption("Tip: upload one picture per line. The voice reads the lines while your pictures show.")

    # Get the script lines from whatever source the user chose
    if mode.startswith("Easy"):
        lines = parse_script_lines(st.session_state["yt_script"]) if "yt_script" in st.session_state else None
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
            "en-IN-PrabhatNeural (India man)"], key="yt_voice_sel")
        voice_code = voice_label.split(" ")[0]

        if st.button("Make voice", key="yt_voice_btn"):
            with st.spinner("AI is speaking... please wait"):
                spoken_text = ". ".join(lines)
                voice_file = make_voice(spoken_text, voice=voice_code)
                st.session_state["yt_voice"] = voice_file
            st.success("Voice ready! Play it below.")
            st.audio(open(voice_file, "rb").read(), format="audio/mp3")

        st.markdown("### Pictures")
        if mode.startswith("Easy"):
            if st.button("Make pictures", key="yt_pics_btn"):
                with st.spinner("AI is drawing the scenes... this can take 1-3 minutes"):
                    progress = st.progress(0)
                    images = []
                    for i, line in enumerate(lines):
                        prompt = (f"YouTube Shorts scene, cinematic photo style: {line}. "
                                  f"Vibrant colors, dramatic lighting, NO text on the image")
                        img_bytes, err = ask_scene_image(prompt)
                        if err:
                            st.error(f"Scene {i+1} failed ({err}). Try again in a minute.")
                            break
                        images.append(img_bytes)
                        progress.progress((i + 1) / len(lines))
                    progress.empty()
                    if images:
                        st.session_state["yt_images"] = images

        # Pictures from whatever source the user chose
        if mode.startswith("Easy"):
            images = st.session_state["yt_images"] if "yt_images" in st.session_state else None
        else:
            images = [f.getvalue() for f in my_files] if my_files else None

        if images:
            st.success(f"{len(images)} scene pictures ready!")
            cols = st.columns(3)
            for i, img in enumerate(images):
                with cols[i % 3]:
                    st.image(img, use_container_width=True)

        st.markdown("### Video")
        if st.button("Make video", key="yt_video_btn", type="primary"):
            if not images:
                st.warning("Make pictures first!" if mode.startswith("Easy")
                           else "Upload at least one picture first!")
            else:
                with st.spinner("Putting voice + pictures together... this takes 1-2 minutes"):
                    # No voice yet? Make one quietly with the chosen voice.
                    if "yt_voice" not in st.session_state:
                        spoken_text = ". ".join(lines)
                        st.session_state["yt_voice"] = make_voice(spoken_text, voice=voice_code)
                    video_file = make_reels_video(images, st.session_state["yt_voice"],
                                                  out_file="shorts.mp4")
                    st.session_state["yt_video"] = video_file
                st.success("Your YouTube Shorts video is ready!")
                st.video(open(video_file, "rb").read(), format="video/mp4")
                st.download_button("Download Shorts (.mp4)",
                                   open(video_file, "rb").read(),
                                   file_name="my_shorts.mp4", mime="video/mp4")

elif page == "Video Editing":
    st.markdown('<div class="main-title">Video Editing</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Upload your video, cut the part you want, add an AI voice-over, '
                'and make it vertical for Reels and Shorts.</div>',
                unsafe_allow_html=True)

    video_file = st.file_uploader("Upload your video (MP4 or MOV)", type=["mp4", "mov"],
                                  key="ve_file")
    if video_file is not None:
        st.video(video_file, format="video/mp4")

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.write(video_file.getvalue())
        tmp.close()

        # Remember which video we are working on (a new upload resets it)
        if st.session_state.get("ve_upload_name") != video_file.name:
            st.session_state["ve_current"] = tmp.name
            st.session_state["ve_upload_name"] = video_file.name
            st.session_state.pop("ve_cut_done", None)

        try:
            dur = get_video_duration(tmp.name)
        except Exception:
            dur = None
            st.error("Sorry, I could not read this video. Please try an MP4 file.")

        if dur:
            st.write(f"Your video is **{dur:.1f} seconds** long.")
            start, end = st.slider("Keep this part (move the two dots)",
                                   0.0, float(dur), (0.0, float(dur)), step=0.5,
                                   key="ve_slider")
            st.write(f"New video: from **{start:.1f}s** to **{end:.1f}s** "
                     f"= **{end - start:.1f} seconds**")

            if st.button("Cut and download", key="ve_cut_btn", type="primary"):
                if end - start < 0.5:
                    st.warning("Pick a bigger part — the two dots are too close together.")
                else:
                    with st.spinner("Cutting your video... this can take a minute"):
                        out = cut_video(tmp.name, start, end, out_file="edited.mp4")
                    st.session_state["ve_current"] = out
                    st.session_state["ve_cut_done"] = True
                    st.success("Your cut video is ready!")
                    st.video(out, format="video/mp4")
                    st.download_button("Download edited video",
                                       open(out, "rb").read(),
                                       file_name="edited.mp4", mime="video/mp4")

        st.markdown("### Voice-over")
        if st.session_state.get("ve_cut_done"):
            st.caption("The voice will be added to your CUT video above.")
        else:
            st.caption("The voice will be added to your full video.")
        voice_words = st.text_area("What should the voice say?",
                                   height=110, key="ve_words",
                                   placeholder="Example: Hello everyone, welcome to my video! Today I will show you...")
        voice_label = st.selectbox("Choose a voice", [
            "en-US-JennyNeural (US woman)",
            "en-US-GuyNeural (US man)",
            "en-IN-NeerjaNeural (India woman)",
            "en-IN-PrabhatNeural (India man)"], key="ve_voice_sel")
        voice_code = voice_label.split(" ")[0]
        keep_sound = st.radio("What about the video's own sound?", [
            "Only my voice (video goes silent)",
            "Keep the video sound quiet under my voice"], key="ve_keep_radio")

        if st.button("Make voice", key="ve_voice_btn"):
            if not voice_words.strip():
                st.warning("Type what the voice should say first!")
            else:
                with st.spinner("AI is speaking... please wait"):
                    vf = make_voice(voice_words.strip(), voice=voice_code)
                    st.session_state["ve_voice"] = vf
                st.success("Voice ready! Play it below.")
                st.audio(open(vf, "rb").read(), format="audio/mp3")

        if st.button("Add voice to video", key="ve_addvoice_btn", type="primary"):
            if "ve_voice" not in st.session_state:
                st.warning("Click 'Make voice' first!")
            else:
                with st.spinner("Adding voice to your video... this can take 1-2 minutes"):
                    out2 = add_voice_to_video(
                        st.session_state["ve_current"], st.session_state["ve_voice"],
                        out_file="edited_with_voice.mp4",
                        keep_original=(keep_sound.startswith("Keep")))
                st.session_state["ve_current"] = out2
                st.success("Your video with voice-over is ready!")
                st.video(out2, format="video/mp4")
                st.download_button("Download video with voice",
                                   open(out2, "rb").read(),
                                   file_name="edited_with_voice.mp4", mime="video/mp4")

        st.markdown("### Make it vertical (for Reels & Shorts)")
        st.caption("Cuts the middle 9:16 piece of your current video and makes it 1080x1920. "
                   "If you already cut it or added a voice, this uses that version.")
        if st.button("Make vertical video", key="ve_vert_btn", type="primary"):
            with st.spinner("Making your video vertical... this can take a minute"):
                out3 = make_vertical(st.session_state["ve_current"], out_file="vertical.mp4")
            st.session_state["ve_current"] = out3
            st.success("Your vertical video is ready!")
            st.video(out3, format="video/mp4")
            st.download_button("Download vertical video", open(out3, "rb").read(),
                               file_name="vertical.mp4", mime="video/mp4")

else:
    # Placeholder page for every tool (we build them one by one)
    st.markdown(f'<div class="main-title">{page}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">This tool is not built yet. It comes in the next steps.</div>',
                unsafe_allow_html=True)
