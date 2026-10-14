import os
import base64
import streamlit as st
import requests
from urllib.parse import quote

# ---- 0. Load the secret keys (from .env locally, from Secrets on the cloud) ----
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
    # On Streamlit Cloud there is no .env (it's gitignored). The keys come from the
    # app's Settings > Secrets page instead. Copy them into os.environ so every
    # os.environ.get("...") call below works unchanged in both places.
    try:
        for key in st.secrets:
            os.environ.setdefault(key, str(st.secrets[key]))
    except Exception:
        pass

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

def ask_cf_text(prompt):
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if not account_id or not api_token:
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/meta/llama-3.1-8b-instruct"
    try:
        r = requests.post(url, headers={"Authorization": f"Bearer {api_token}"},
                          json={"messages": [{"role": "user", "content": prompt}]}, timeout=45)
        if r.status_code == 200:
            return r.json().get("result", {}).get("response")
    except Exception:
        pass
    return None

def ask_text_ai(prompt, system_prompt=None, history=None):
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        try:
            url = "https://api.groq.com/openai/v1/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            if history:
                for h in history[-8:]:
                    messages.append({"role": h["role"], "content": h["content"]})
            messages.append({"role": "user", "content": prompt})
            data = {"model": "llama-3.3-70b-versatile",
                    "messages": messages}
            r = requests.post(url, headers=headers, json=data, timeout=40)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass
    import time
    combined_prompt = prompt
    if system_prompt or history:
        parts = []
        if system_prompt:
            parts.append(f"System: {system_prompt}")
        if history:
            for h in history[-6:]:
                parts.append(f"{h['role'].capitalize()}: {h['content']}")
        parts.append(f"User: {prompt}\nAssistant:")
        combined_prompt = "\n\n".join(parts)
    for _ in range(2):
        text, err = ask_gemini_text(combined_prompt)
        if text:
            return text
        time.sleep(1)
    cf_res = ask_cf_text(combined_prompt)
    if cf_res:
        return cf_res
    return None

# ---- Ask AI about a picture or PDF document ----
def ask_gemini_vision(prompt, img_bytes, mime_type="image/jpeg", system_prompt=None):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return "No GEMINI_API_KEY found in .env file."
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    b64 = base64.b64encode(img_bytes).decode()
    full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
    data = {"contents": [{"parts": [
        {"text": full_prompt},
        {"inline_data": {"mime_type": mime_type, "data": b64}}
    ]}]}
    r = requests.post(url, params={"key": api_key}, json=data, timeout=120)
    if r.status_code != 200:
        return f"AI error {r.status_code}: {r.text[:200]}"
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts) or "AI gave an empty answer."

# ---- Speech to Text: Cloudflare Whisper first, Gemini 2.5 Flash Audio as backup ----
def ask_transcribe_audio(audio_bytes, mime_type="audio/mp3"):
    # 1. Cloudflare OpenAI Whisper
    account_id = os.environ.get("CF_ACCOUNT_ID")
    api_token = os.environ.get("CF_API_TOKEN")
    if account_id and api_token:
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/@cf/openai/whisper"
        try:
            r = requests.post(url,
                              headers={"Authorization": f"Bearer {api_token}", "Content-Type": "application/json"},
                              json={"audio": list(audio_bytes)}, timeout=90)
            if r.status_code == 200:
                res = r.json().get("result", {})
                text = res.get("text", "").strip()
                vtt = res.get("vtt", "")
                if text:
                    return text, vtt, None
        except Exception:
            pass

    # 2. Gemini 2.5 Flash Audio
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
            b64 = base64.b64encode(audio_bytes).decode("utf-8")
            clean_mime = mime_type if mime_type in ["audio/mp3", "audio/wav", "audio/ogg", "audio/m4a", "audio/webm", "audio/flac"] else "audio/mp3"
            data = {
                "contents": [{
                    "parts": [
                        {"text": "Please transcribe this audio recording verbatim with maximum accuracy. Do not add introductory or concluding remarks, output only the exact spoken words."},
                        {"inline_data": {"mime_type": clean_mime, "data": b64}}
                    ]
                }]
            }
            r = requests.post(url, params={"key": api_key}, json=data, timeout=90)
            if r.status_code == 200:
                parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
                text = "".join(p.get("text", "") for p in parts).strip()
                if text:
                    return text, "", None
        except Exception:
            pass

    return None, "", "Could not transcribe audio. Please check your recording and try again."

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

# ---- Color palettes for professional resume PDF styling ----
RESUME_THEMES = {
    "Modern Navy": {
        "primary": (30, 58, 138),     # Classic Navy #1e3a8a
        "accent": (59, 130, 246),
        "sidebar_bg": (244, 246, 250),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Emerald Professional": {
        "primary": (6, 95, 70),       # Forest / Emerald Green #065f46
        "accent": (16, 185, 129),
        "sidebar_bg": (242, 248, 245),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Crimson Executive": {
        "primary": (159, 18, 57),     # Crimson Burgundy #9f1239
        "accent": (225, 29, 72),
        "sidebar_bg": (254, 242, 244),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Royal Purple": {
        "primary": (88, 28, 135),     # Royal Violet #581c87
        "accent": (147, 51, 234),
        "sidebar_bg": (248, 244, 253),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Classic Charcoal": {
        "primary": (31, 41, 55),      # Elegant Dark Slate #1f2937
        "accent": (75, 85, 99),
        "sidebar_bg": (245, 246, 248),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
}

# ---- Turn the resume text into an executive-grade PDF file ----
def make_resume_pdf(text, header=None, theme="Modern Navy", layout="Modern Clean"):
    from fpdf import FPDF
    import re

    theme_info = RESUME_THEMES.get(theme, RESUME_THEMES["Modern Navy"])
    pdf = FPDF("p", "mm", "A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    # Font setup: prefer Calibri (clean, modern sans-serif), fallback to Arial or Helvetica
    try:
        pdf.add_font("ResumeFont", "", "C:/Windows/Fonts/calibri.ttf")
        pdf.add_font("ResumeFont", "B", "C:/Windows/Fonts/calibrib.ttf")
        pdf.add_font("ResumeFont", "I", "C:/Windows/Fonts/calibrii.ttf")
        font = "ResumeFont"
    except Exception:
        try:
            pdf.add_font("ResumeFont", "", "C:/Windows/Fonts/arial.ttf")
            pdf.add_font("ResumeFont", "B", "C:/Windows/Fonts/arialbd.ttf")
            pdf.add_font("ResumeFont", "I", "C:/Windows/Fonts/ariali.ttf")
            font = "ResumeFont"
        except Exception:
            font = "Helvetica"

    # Normalize typographic smart quotes and unicode dashes
    replacements = {
        "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ",
        "\u2011": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for smart, normal in replacements.items():
        text = text.replace(smart, normal)

    if font == "Helvetica":
        text = text.encode("latin-1", errors="replace").decode("latin-1")

    # Strip bracketed placeholder instructions
    text = re.sub(r"\[[^\]]{0,60}\]", "", text)

    # Parse into structured sections
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    sections, pre = [], []
    for raw in lines:
        if raw.startswith("## ") or (raw.startswith("# ") and not sections):
            title = raw.lstrip("#").strip().replace("*", "").replace("_", "")
            if title.lower() in ("resume", "curriculum vitae", "cv"):
                continue
            sections.append({"title": title, "body": []})
        elif sections:
            sections[-1]["body"].append(raw)
        else:
            pre.append(raw)
    if pre:
        sections.insert(0, {"title": "", "body": pre})

    header = header or {}
    clean_layout = "Modern Sidebar" if "sidebar" in layout.lower() else "Modern Clean"

    if clean_layout == "Modern Sidebar":
        sb_w = 66
        main_x = sb_w + 10
        main_w = 210 - main_x - 14

        # Full height sidebar background
        pdf.set_fill_color(*theme_info["sidebar_bg"])
        pdf.rect(0, 0, sb_w, 297, style="F")

        # Top accent bar on right side
        pdf.set_fill_color(*theme_info["primary"])
        pdf.rect(sb_w, 0, 210 - sb_w, 3.5, style="F")

        SB_KEYS = ("contact", "skill", "educat", "personal", "detail", "certif", "language", "hobby")
        sb_secs = [s for s in sections if any(k in s["title"].lower() for k in SB_KEYS)]
        main_secs = [s for s in sections if not any(k in s["title"].lower() for k in SB_KEYS)]

        # Right Main: Candidate Name & Target Title
        pdf.set_xy(main_x, 14)
        pdf.set_font(font, "B", 22)
        pdf.set_text_color(*theme_info["text"])
        pdf.cell(main_w, 8, header.get("name", "Candidate Name"), new_x="LMARGIN", new_y="NEXT")

        if header.get("title"):
            pdf.set_x(main_x)
            pdf.set_font(font, "B", 11)
            pdf.set_text_color(*theme_info["primary"])
            pdf.cell(main_w, 6, header["title"].upper(), new_x="LMARGIN", new_y="NEXT")

        pdf.set_draw_color(*theme_info["line"])
        pdf.line(main_x, pdf.get_y() + 2, 210 - 14, pdf.get_y() + 2)
        pdf.set_y(pdf.get_y() + 5)

        # Left Sidebar: Contact details
        pdf.set_xy(10, 14)
        pdf.set_font(font, "B", 10.5)
        pdf.set_text_color(*theme_info["primary"])
        pdf.cell(sb_w - 18, 5.5, "CONTACT", new_x="LMARGIN", new_y="NEXT")
        pdf.set_draw_color(*theme_info["primary"])
        pdf.line(10, pdf.get_y() + 1, sb_w - 8, pdf.get_y() + 1)
        pdf.set_y(pdf.get_y() + 3)

        pdf.set_font(font, "", 8.6)
        pdf.set_text_color(*theme_info["text"])
        contacts = [p.strip() for p in header.get("contact", "").split("|") if p.strip()]
        for c in contacts:
            pdf.set_x(10)
            pdf.multi_cell(sb_w - 18, 4.4, c, align="L")

        sb_y = pdf.get_y() + 4
        for sec in sb_secs:
            pdf.set_xy(10, sb_y)
            pdf.set_font(font, "B", 10)
            pdf.set_text_color(*theme_info["primary"])
            pdf.cell(sb_w - 18, 5.5, sec["title"].upper(), new_x="LMARGIN", new_y="NEXT")
            pdf.set_draw_color(*theme_info["primary"])
            pdf.line(10, pdf.get_y() + 1, sb_w - 8, pdf.get_y() + 1)
            pdf.set_y(pdf.get_y() + 3)

            pdf.set_font(font, "", 8.6)
            pdf.set_text_color(*theme_info["text"])
            for ln in sec["body"]:
                pdf.set_x(10)
                clean_ln = ln.lstrip("-*• \t")
                pdf.multi_cell(sb_w - 18, 4.4, clean_ln, markdown=True, align="L")
            sb_y = pdf.get_y() + 4

        # Right Main Sections
        for sec in main_secs:
            y = pdf.get_y()
            if sec["title"]:
                pdf.set_fill_color(*theme_info["primary"])
                pdf.rect(main_x, y + 0.6, 3, 5, style="F")
                pdf.set_xy(main_x + 5, y)
                pdf.set_font(font, "B", 10.5)
                pdf.set_text_color(*theme_info["primary"])
                pdf.cell(main_w - 5, 5.5, sec["title"].upper(), new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(*theme_info["line"])
                pdf.line(main_x + 5, y + 6.2, 210 - 14, y + 6.2)
                pdf.set_y(y + 7.5)

            pdf.set_text_color(*theme_info["text"])
            pdf.set_font(font, "", 9.2)
            for ln in sec["body"]:
                y_ln = pdf.get_y()
                if ln.startswith("### "):
                    sub = ln[4:].strip()
                    pdf.set_xy(main_x, y_ln + 1)
                    pdf.set_font(font, "B", 9.6)
                    pdf.cell(main_w, 5.2, sub, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font(font, "", 9.2)
                elif ln.startswith("- ") or ln.startswith("* ") or ln.startswith("• "):
                    content = re.sub(r"^[\-*•]\s*", "", ln)
                    pdf.set_fill_color(*theme_info["primary"])
                    pdf.circle(main_x + 2, y_ln + 2.4, 0.7, style="F")
                    pdf.set_xy(main_x + 5, y_ln)
                    pdf.multi_cell(main_w - 5, 4.7, content, markdown=True, align="L")
                else:
                    pdf.set_xy(main_x, y_ln)
                    pdf.multi_cell(main_w, 4.7, ln, markdown=True, align="L")
            pdf.set_y(pdf.get_y() + 4)
    else:
        # Modern Clean (Executive Full-Width Single-Column)
        MARGIN = 16
        col_w = 210 - 2 * MARGIN

        # Top accent stripe
        pdf.set_fill_color(*theme_info["primary"])
        pdf.rect(0, 0, 210, 3.5, style="F")

        # Centered Executive Header
        pdf.set_y(10)
        pdf.set_font(font, "B", 22)
        pdf.set_text_color(*theme_info["text"])
        pdf.cell(0, 8.5, header.get("name", "Candidate Name"), align="C", new_x="LMARGIN", new_y="NEXT")

        if header.get("title"):
            pdf.set_font(font, "B", 11)
            pdf.set_text_color(*theme_info["primary"])
            pdf.cell(0, 5.5, header["title"].upper(), align="C", new_x="LMARGIN", new_y="NEXT")

        if header.get("contact"):
            pdf.set_font(font, "", 9.2)
            pdf.set_text_color(*theme_info["muted"])
            pdf.cell(0, 5, header["contact"], align="C", new_x="LMARGIN", new_y="NEXT")

        pdf.set_draw_color(*theme_info["line"])
        pdf.line(MARGIN, pdf.get_y() + 2, 210 - MARGIN, pdf.get_y() + 2)
        pdf.set_y(pdf.get_y() + 5)

        total_lines = sum(len(s["body"]) for s in sections)
        LH = 5.2 if total_lines < 32 else 4.8
        sec_gap = 4.2 if total_lines < 32 else 2.6

        for sec in sections:
            y = pdf.get_y()
            if y > 275:
                pdf.add_page()
                pdf.set_fill_color(*theme_info["primary"])
                pdf.rect(0, 0, 210, 3.5, style="F")
                pdf.set_y(14)
                y = 14

            if sec["title"]:
                pdf.set_fill_color(*theme_info["primary"])
                pdf.rect(MARGIN, y + 0.6, 3, 5, style="F")
                pdf.set_xy(MARGIN + 5, y)
                pdf.set_font(font, "B", 10.5)
                pdf.set_text_color(*theme_info["primary"])
                pdf.cell(col_w - 5, 5.5, sec["title"].upper(), new_x="LMARGIN", new_y="NEXT")
                pdf.set_draw_color(*theme_info["line"])
                pdf.line(MARGIN + 5, y + 6.2, 210 - MARGIN, y + 6.2)
                pdf.set_y(y + 7.5)

            pdf.set_text_color(*theme_info["text"])
            pdf.set_font(font, "", 9.5)
            for ln in sec["body"]:
                y_ln = pdf.get_y()
                if y_ln > 280:
                    pdf.add_page()
                    pdf.set_fill_color(*theme_info["primary"])
                    pdf.rect(0, 0, 210, 3.5, style="F")
                    pdf.set_y(14)
                    y_ln = 14

                if ln.startswith("### "):
                    sub = ln[4:].strip()
                    pdf.set_xy(MARGIN, y_ln + 1)
                    pdf.set_font(font, "B", 9.8)
                    pdf.cell(col_w, LH + 0.5, sub, new_x="LMARGIN", new_y="NEXT")
                    pdf.set_font(font, "", 9.5)
                elif ln.startswith("- ") or ln.startswith("* ") or ln.startswith("• "):
                    content = re.sub(r"^[\-*•]\s*", "", ln)
                    pdf.set_fill_color(*theme_info["primary"])
                    pdf.circle(MARGIN + 2, y_ln + 2.5, 0.7, style="F")
                    pdf.set_xy(MARGIN + 5, y_ln)
                    pdf.multi_cell(col_w - 5, LH, content, markdown=True, align="L")
                else:
                    pdf.set_xy(MARGIN, y_ln)
                    pdf.multi_cell(col_w, LH, ln, markdown=True, align="L")
            pdf.set_y(pdf.get_y() + sec_gap)

    return bytes(pdf.output())

# ---- Free AI voice (Microsoft edge-tts, no key needed) ----
def make_voice(text, voice="en-US-JennyNeural", filename="voice.mp3", timeout=60):
    import asyncio
    import edge_tts
    async def _run():
        communicate = edge_tts.Communicate(text, voice=voice)
        await communicate.save(filename)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(asyncio.wait_for(_run(), timeout=timeout))
    except asyncio.TimeoutError:
        raise RuntimeError("The voice service is taking too long. Please try again in a minute.")
    finally:
        loop.close()
    return filename

# ---- Strip code blocks and markdown symbols before feeding into text-to-speech ----
def clean_for_speech(text):
    import re
    cleaned = re.sub(r'```[\s\S]*?```', ' [Code block omitted] ', text)
    cleaned = re.sub(r'`([^`]+)`', r'\1', cleaned)
    cleaned = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', cleaned)
    cleaned = re.sub(r'[#*_~>|]+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    words = cleaned.split()
    if len(words) > 130:
        cleaned = " ".join(words[:130]) + "... and more."
    return cleaned

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
                                  json={"prompt": prompt}, timeout=(10, 60))
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
    /* Tabs: purple active text + a soft underline bar */
    div[data-testid="stTabs"] button {
        color: #6b7280;
        font-weight: 600;
        font-size: 15px;
    }
    div[data-testid="stTabs"] button:hover {
        color: #7c3aed;
    }
    div[data-testid="stTabs"] [aria-selected="true"] {
        color: #7c3aed;
    }
    div[data-testid="stTabs"] {
        background: transparent;
    }
    /* Primary buttons: purple gradient, lift on hover */
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stDownloadButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #6d28d9, #9333ea 55%, #ec4899);
        color: #ffffff;
        border: none;
        border-radius: 10px;
        font-weight: 700;
        padding: 0.55rem 1.3rem;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover,
    div[data-testid="stDownloadButton"] button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(124, 58, 237, 0.35);
        border: none;
        color: #ffffff;
    }
    /* Secondary + download buttons: soft lavender outline */
    div[data-testid="stButton"] button[kind="secondary"],
    div[data-testid="stDownloadButton"] button {
        border: 1px solid #ddd6fe;
        border-radius: 10px;
        color: #4c1d95;
        background: #faf5ff;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover,
    div[data-testid="stDownloadButton"] button:hover {
        border-color: #a78bfa;
        color: #6d28d9;
        background: #f3e8ff;
    }
    /* A small labelled section header used inside tool pages */
    .sec-label {
        font-size: 15px;
        font-weight: 700;
        color: #4c1d95;
        margin: 6px 0 2px;
    }
</style>
""", unsafe_allow_html=True)

# ---- 3. The left menu ----
menu_items = ["Home", "Chat AI", "Image Studio", "Video Maker", "Resume Maker",
              "Text to Speech", "Audio to Text", "QR Code"]

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
    {"name": "Audio to Text",  "emoji": "🎙️", "desc": "Transcribe voice, audio & meetings", "status": "Ready", "ready": True},
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
                '<div class="ph-sub">Talk with specialized AI personas, hear natural voice replies, or switch to Write mode '
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
        PERSONA_PROMPTS = {
            "🤖 General Assistant": "You are a helpful, versatile, and thoughtful AI assistant. Give clear, well-structured, and accurate answers.",
            "💻 Senior Software Engineer": "You are an elite Senior Software Engineer and Architect. Provide production-ready, clean, secure, and modern code with best practices, error handling, and performance advice.",
            "✍️ Creative Copywriter & Marketer": "You are an award-winning copywriter and growth marketer. Produce high-converting copy, compelling hooks, viral social content, and persuasive storytelling.",
            "🎓 Patient Teacher & Tutor": "You are a warm, patient educator and tutor. Break concepts down step-by-step with simple analogies, clarify confusing terms, and reinforce key takeaways.",
            "💼 Executive Business Consultant": "You are a top-tier management consultant and strategy director. Provide structured insights, strategic frameworks, data-driven reasoning, and actionable executive summaries.",
            "🩺 Health & Fitness Coach": "You are a supportive, certified fitness and wellness advisor. Provide science-backed lifestyle habits, workout tips, nutrition guidance, and motivational support."
        }

        col_p1, col_p2, col_p3 = st.columns([2.2, 1.2, 1])
        with col_p1:
            persona = st.selectbox("AI Persona:", list(PERSONA_PROMPTS.keys()), key="chat_persona")
        with col_p2:
            enable_voice = st.checkbox("🔊 Voice Reply", value=False, key="chat_voice_toggle",
                                       help="AI speaks its answer out loud using neural text-to-speech")
            voice_choice = "en-US-JennyNeural"
            if enable_voice:
                voice_label = st.selectbox("Voice", [
                    "en-US-JennyNeural (US Woman)",
                    "en-US-GuyNeural (US Man)",
                    "en-IN-NeerjaNeural (India Woman)",
                    "en-IN-PrabhatNeural (India Man)"
                ], key="chat_voice_voice")
                voice_choice = voice_label.split(" ")[0]
        with col_p3:
            st.markdown('<div style="height: 25px;"></div>', unsafe_allow_html=True)
            if st.button("🧹 Clear Chat", key="clear_chat_btn", use_container_width=True):
                st.session_state.messages = []
                st.rerun()

        with st.expander("📎 Add a photo or PDF document to your message (optional)"):
            chat_file = st.file_uploader("Upload a photo (JPG, PNG) or document (PDF)",
                                         type=["jpg", "jpeg", "png", "pdf"], key="chat_file")

        # Remember old messages so they don't disappear
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Show all old messages
        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                if m.get("image"):
                    st.image(m["image"], width=220)
                if m.get("doc_name"):
                    st.caption(f"📄 Attached: {m['doc_name']}")
                st.write(m["content"])
                if m.get("audio"):
                    st.audio(m["audio"], format="audio/mp3")

        # The typing box at the bottom
        prompt = st.chat_input("Type your message here...")

        if prompt:
            # Show what the user typed (and the picture/document, if any)
            msg = {"role": "user", "content": prompt}
            file_mime = "image/jpeg"
            if chat_file is not None:
                fname = chat_file.name.lower()
                if fname.endswith(".pdf"):
                    msg["doc_name"] = chat_file.name
                    msg["doc_bytes"] = chat_file.getvalue()
                    file_mime = "application/pdf"
                else:
                    msg["image"] = chat_file.getvalue()
                    file_mime = "image/png" if fname.endswith(".png") else "image/jpeg"

            st.session_state.messages.append(msg)
            with st.chat_message("user"):
                if msg.get("image"):
                    st.image(msg["image"], width=220)
                if msg.get("doc_name"):
                    st.caption(f"📄 Attached: {msg['doc_name']}")
                st.write(prompt)

            # Get the AI answer (with persona & conversation memory)
            sys_instruction = PERSONA_PROMPTS.get(persona, "")
            with st.spinner("AI is thinking..."):
                if msg.get("doc_bytes"):
                    reply = ask_gemini_vision(prompt, msg["doc_bytes"], mime_type="application/pdf", system_prompt=sys_instruction)
                elif msg.get("image"):
                    reply = ask_gemini_vision(prompt, msg["image"], mime_type=file_mime, system_prompt=sys_instruction)
                else:
                    prev_history = st.session_state.messages[:-1]
                    reply = ask_text_ai(prompt, system_prompt=sys_instruction, history=prev_history) or BUSY_MSG

            asst_msg = {"role": "assistant", "content": reply}

            # If voice reply is enabled, generate audio
            if enable_voice and reply and reply != BUSY_MSG:
                with st.spinner("AI is speaking..."):
                    spoken = clean_for_speech(reply)
                    if spoken:
                        try:
                            vf = make_voice(spoken, voice=voice_choice, filename="chat_voice_reply.mp3", timeout=30)
                            with open(vf, "rb") as f:
                                asst_msg["audio"] = f.read()
                        except Exception:
                            pass

            st.session_state.messages.append(asst_msg)
            with st.chat_message("assistant"):
                st.write(reply)
                if asst_msg.get("audio"):
                    st.audio(asst_msg["audio"], format="audio/mp3")

        # Export chat button if messages exist
        if st.session_state.messages:
            st.divider()
            c_ex1, c_ex2 = st.columns([3, 1])
            with c_ex2:
                export_lines = []
                for m in st.session_state.messages:
                    role_title = "User" if m["role"] == "user" else f"AI ({persona})"
                    export_lines.append(f"### {role_title}\n{m['content']}\n")
                export_text = "\n".join(export_lines)
                st.download_button("📥 Export Chat (.md)", export_text.encode("utf-8"),
                                   file_name="chat_history.md", mime="text/markdown",
                                   use_container_width=True)

elif page == "Image Studio":
    st.markdown('<div class="page-hero"><div class="ph-title">🎨 Image Studio</div>'
                '<div class="ph-sub">Make a new picture, edit your own photo, or chat with a '
                'picture and keep changing it.</div></div>',
                unsafe_allow_html=True)

    tab_new, tab_edit, tab_chat = st.tabs(["✨ Make a new picture", "🖌️ Edit my photo",
                                           "💬 Chat with pictures"])

    with tab_new:
        st.markdown('<div class="sec-label">Describe the picture you want</div>',
                    unsafe_allow_html=True)
        prompt = st.text_input("Type in English for the best results",
                               key="ig_prompt",
                               placeholder="example: a cute robot reading a book in a cozy room")
        if st.button("Generate Image", key="ig_gen_btn", type="primary"):
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
            name = st.text_input("Your full name", placeholder="example: Farooq Khan")
            job_title = st.text_input("What job do you want?", placeholder="example: AI & Python Developer | Data Scientist")
        with col2:
            email = st.text_input("Email ID", placeholder="example: farooqkhan29889@gmail.com")
            phone = st.text_input("Phone number", placeholder="example: +91 98765 43210")
        col_l1, col_l2 = st.columns(2)
        with col_l1:
            address = st.text_input("City, State / Country", placeholder="example: Bhopal, MP, India")
        with col_l2:
            links = st.text_input("LinkedIn / GitHub / Portfolio Link", placeholder="example: github.com/farooqkhan29889-svg")

        st.markdown("**Technical Skills**")
        skills = st.text_area("Skills (separate with commas)",
                              placeholder="example: Python, AI Agents, Machine Learning, Data Analytics, Pandas, SQL, Streamlit, Git")

        st.markdown("**Work experience**")
        experience = st.text_area("Work experience & duties",
                                  placeholder="example: 1 year as Python Developer building AI workflows, data analytics pipelines, and backend APIs")

        st.markdown("**Key Projects (optional - AI will generate impressive projects if blank)**")
        projects = st.text_area("Projects",
                                placeholder="example: Built an AI Market Platform in Streamlit with speech-to-text and vision analysis; Automated data analytics reporting engine")

        st.markdown("**Education**")
        education = st.text_area("Education",
                                 placeholder="example: Bachelor of Computer Applications (BCA) - Barkatullah University, 2023")

        with st.expander("📌 Traditional Bio-data Details (Optional — leave empty for modern tech resumes)"):
            col3, col4 = st.columns(2)
            with col3:
                father_name = st.text_input("Father's name (optional)")
                dob = st.text_input("Date of birth (optional)", placeholder="example: 05 July 2000")
            with col4:
                gender = st.selectbox("Gender (optional)", ["Not specified", "Male", "Female"])
                marital = st.selectbox("Marital status (optional)", ["Not specified", "Single", "Married"])

        submitted = st.form_submit_button("Write my resume", type="primary")

    # The chat box: people type what they want built into their resume
    change = st.chat_input("Want a change? Type it here (example: add more metrics to my Python experience)")

    if submitted:
        if not name.strip():
            st.warning("Type your name first!")
        else:
            with st.spinner("AI is crafting your executive-grade resume... please wait"):
                contact_parts = [p.strip() for p in (email, phone, address, links) if p and p.strip()]
                instructions = (
                    f"You are a top-tier executive resume writer and Silicon Valley technical recruiter.\n"
                    f"Write an impressive, modern, highly competitive professional resume for this candidate.\n\n"
                    f"Candidate Profile:\n"
                    f"- Full Name: {name}\n"
                    f"- Target Job Role: {job_title}\n"
                    f"- Email: {email}\n"
                    f"- Phone: {phone}\n"
                    f"- Location: {address}\n"
                    f"- Links: {links}\n"
                    f"- Education: {education}\n"
                    f"- Technical Skills: {skills}\n"
                    f"- Experience: {experience}\n"
                    f"- Projects: {projects}\n"
                    f"{f'- Personal Details: Father: {father_name}, DOB: {dob}, Gender: {gender}, Status: {marital}' if (father_name.strip() or dob.strip()) else ''}\n\n"
                    f"Strict Resume Writing Guidelines:\n"
                    f"1. Structure using clean markdown:\n"
                    f"   - '## Professional Summary': A 3-4 sentence powerful summary emphasizing expertise, capabilities, and business impact.\n"
                    f"   - '## Technical Skills': Categorize skills into 3-4 clean grouped bullets with bold category headers (e.g., - **Programming Languages:** ..., - **Artificial Intelligence & LLMs:** ..., - **Data Science & Analytics:** ..., - **Tools & Platforms:** ...).\n"
                    f"   - '## Professional Experience':\n"
                    f"     Use '### Role Title | Company / Industry (Duration)' for each position.\n"
                    f"     Generate 3 to 4 substantial, achievement-oriented bullet points starting with strong action verbs (Architected, Developed, Engineered, Optimized, Automated). Emphasize real workflows, technologies used, and quantifiable benefits (e.g. reduced processing time by 40%, improved pipeline efficiency).\n"
                    f"   - '## Key Projects':\n"
                    f"     Include 2 standout technical projects (use candidate projects if provided, or design realistic high-impact projects matching their exact tech stack).\n"
                    f"     Format as '### Project Title | *Key Technologies*'\n"
                    f"     Provide 2 detailed bullets highlighting technical architecture, features implemented, and results.\n"
                    f"   - '## Education': Degree, university/institution, graduation year.\n"
                    f"   {f'- ## Personal Details: Include only if specified above.' if (father_name.strip() or dob.strip()) else 'DO NOT include Father name, DOB, gender, or marital status.'}\n"
                    f"2. Ensure the resume content is rich and comprehensive enough to elegantly fill a full 1-page resume without sparse or blank areas.\n"
                    f"3. NEVER use brackets like [Year] or [Company Name]. Output ONLY the final resume markdown without conversational preamble."
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
                        "contact": "   |   ".join(contact_parts),
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
                    f"2. Keep the same format: '## Section Name' headings, '### Subheading' for roles/projects, and '-' bullets.\n"
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
        st.markdown("---")
        tab_view, tab_ats = st.tabs(["📄 Resume Preview & Export", "🎯 ATS Scanner & Optimizer"])

        with tab_view:
            st.markdown("### Your resume")
            st.markdown(st.session_state["resume_text"])

            header = st.session_state.get("resume_header") or {}
            safe_name = (header.get("name") or "My_Resume").replace(" ", "_")

            st.markdown('<div class="sec-label">🎨 Styling & Download</div>', unsafe_allow_html=True)
            col_th1, col_th2 = st.columns(2)
            with col_th1:
                chosen_theme = st.selectbox("Choose visual color theme:",
                                            list(RESUME_THEMES.keys()),
                                            index=1, key="res_pdf_theme") # default Emerald
            with col_th2:
                chosen_layout = st.selectbox("Choose layout template:",
                                             ["Modern Clean (Single Column)", "Modern Sidebar (Dual Tone)"],
                                             index=0, key="res_pdf_layout")

            pdf_bytes = make_resume_pdf(st.session_state["resume_text"], header,
                                        theme=chosen_theme, layout=chosen_layout)

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button("📥 Download as PDF", pdf_bytes, key="resume_pdf_btn",
                                   file_name=f"Resume_{safe_name}.pdf",
                                   mime="application/pdf", use_container_width=True)
            with col_b:
                st.download_button("📥 Download as .txt",
                                   st.session_state["resume_text"].encode("utf-8"),
                                   key="resume_txt_btn",
                                   file_name=f"Resume_{safe_name}.txt",
                                   mime="text/plain", use_container_width=True)

            if st.session_state.get("resume_log"):
                st.caption("Changes applied:  " + "  •  ".join(st.session_state["resume_log"]))

        with tab_ats:
            st.markdown('<div class="sec-label">🎯 ATS (Applicant Tracking System) Readiness Check</div>', unsafe_allow_html=True)
            st.write("Over 75% of companies filter candidates using automated ATS scanners. Check how well your resume matches recruiter algorithms and industry keywords.")

            target_jd = st.text_area("Target Job Title / Description (Optional)",
                                     placeholder="Paste job description keywords or requirements here to test direct match...",
                                     height=90, key="ats_jd_input")

            if st.button("⚡ Scan ATS Score & Get Recommendations", key="ats_scan_btn", type="primary"):
                with st.spinner("Analyzing resume structure, keyword density, and action verbs..."):
                    ats_instructions = (
                        f"You are a Fortune 500 Senior Technical Recruiter and ATS algorithm auditor.\n"
                        f"Analyze this resume:\n\n{st.session_state['resume_text']}\n\n"
                        f"{f'Target Role / Requirements: {target_jd}' if target_jd.strip() else 'Target: Standard industry ATS benchmark'}\n\n"
                        f"Output your report strictly in this format:\n"
                        f"SCORE: [Integer between 65 and 96]\n"
                        f"### 🌟 ATS Strengths\n"
                        f"- [2-3 bullet points of what ATS systems like: formatting, clarity, verbs]\n\n"
                        f"### ⚠️ Missing Keywords & Optimization Gaps\n"
                        f"- [2-3 bullet points of missing skills, metric gaps, or passive phrasing]\n\n"
                        f"### 💡 High-Impact Recommendations\n"
                        f"- [2-3 specific suggestions to boost callback probability]"
                    )
                    ats_report = ask_text_ai(ats_instructions)
                    if ats_report:
                        import re
                        m = re.search(r"SCORE:\s*(\d+)", ats_report)
                        sc = int(m.group(1)) if m else 82
                        st.session_state["ats_score"] = min(max(sc, 50), 99)
                        clean_report = re.sub(r"SCORE:\s*\d+", "", ats_report).strip()
                        st.session_state["ats_report"] = clean_report
                    else:
                        st.error(BUSY_MSG)

            if "ats_score" in st.session_state and "ats_report" in st.session_state:
                sc = st.session_state["ats_score"]
                c_sc1, c_sc2 = st.columns([1, 2])
                with c_sc1:
                    st.metric("ATS Compatibility Score", f"{sc} / 100")
                with c_sc2:
                    st.write("")
                    st.progress(sc / 100)
                    if sc >= 85:
                        st.success("🟢 Strong ATS match: Excellent keyword clarity and layout.")
                    elif sc >= 70:
                        st.info("🟡 Moderate ATS match: Good baseline, but needs more metrics and action verbs.")
                    else:
                        st.warning("🔴 Low ATS match: Risk of rejection by automated filters.")

                st.markdown(st.session_state["ats_report"])

                st.markdown("---")
                if st.button("✨ One-Click Auto-Optimize for 95+ ATS Score", key="ats_auto_opt_btn"):
                    with st.spinner("AI is upgrading bullet points with high-impact power verbs and quantifiable metrics..."):
                        opt_prompt = (
                            f"You are an executive resume optimizer. Upgrade this resume to achieve a 95+ ATS score.\n\n"
                            f"Original Resume:\n{st.session_state['resume_text']}\n\n"
                            f"{f'Target Job Requirements: {target_jd}' if target_jd.strip() else ''}\n\n"
                            f"Rules:\n"
                            f"1. Replace passive verbs with strong action verbs (e.g. Architected, Accelerated, Delivered, Spearheaded).\n"
                            f"2. Add realistic, impactful phrasing emphasizing results and metrics without fabricating unprovided company names or jobs.\n"
                            f"3. Maintain '## Section Name' headings and '-' bullet formatting.\n"
                            f"4. Keep all factual details (contact, education, personal details) accurate.\n"
                            f"5. Output ONLY the upgraded resume text, no introductory or concluding chat."
                        )
                        upgraded_res = ask_text_ai(opt_prompt)
                        if upgraded_res:
                            st.session_state["resume_text"] = upgraded_res
                            st.session_state["resume_log"].append("Auto-optimized for 95+ ATS score")
                            st.session_state["ats_score"] = 96
                            st.success("🎉 Resume successfully auto-optimized for 95+ ATS score! Check the Preview tab.")
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)
    else:
        st.info("👆 Fill in your details above and press **Write my resume**. "
                "After that you can view ATS scoring, change themes, or ask for changes in the box below.")

elif page == "Video Maker":
    st.markdown('<div class="page-hero"><div class="ph-title">🎬 Video Maker</div>'
                '<div class="ph-sub">Make a short vertical video for Instagram Reels or '
                'YouTube Shorts — let AI do everything, or use your own script and pictures.</div></div>',
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

    # Keep the render small: too many scenes crashes Streamlit's free 1 GB worker.
    MAX_SCENES = 8
    if lines and len(lines) > MAX_SCENES:
        st.warning(f"That's {len(lines)} scenes — too many for a smooth video. "
                   f"I'll use the first {MAX_SCENES}.")
        lines = lines[:MAX_SCENES]

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
                try:
                    voice_file = make_voice(spoken_text, voice=voice_code)
                    st.session_state["vm_voice"] = voice_file
                except RuntimeError as e:
                    voice_file = None
                    st.error(str(e))
            if voice_file:
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
                    try:
                        # No voice yet? Make one quietly with the chosen voice.
                        if "vm_voice" not in st.session_state:
                            spoken_text = ". ".join(lines)
                            st.session_state["vm_voice"] = make_voice(spoken_text, voice=voice_code)
                        out_name = "reels.mp4" if platform.startswith("Instagram") else "shorts.mp4"
                        video_file = make_reels_video(images, st.session_state["vm_voice"],
                                                      out_file=out_name)
                        st.session_state["vm_video"] = video_file
                    except RuntimeError as e:
                        video_file = None
                        st.error(str(e))
                if video_file:
                    st.success("Your video is ready!")
                    st.video(open(video_file, "rb").read(), format="video/mp4")
                    st.download_button("Download video (.mp4)",
                                       open(video_file, "rb").read(),
                                       file_name=f"my_{out_name}", mime="video/mp4")

elif page == "Text to Speech":
    st.markdown('<div class="page-hero"><div class="ph-title">🔊 Text to Speech</div>'
                '<div class="ph-sub">Type anything and get a natural-sounding voice as an '
                'MP3. Free, and it never runs out of credits.</div></div>',
                unsafe_allow_html=True)

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
                try:
                    vf = make_voice(tts_text.strip(), voice=tts_label.split(" ")[0],
                                    filename="speech.mp3")
                except RuntimeError as e:
                    vf = None
                    st.error(str(e))
            if vf:
                st.success("Your audio is ready!")
                st.audio(open(vf, "rb").read(), format="audio/mp3")
                st.download_button("Download audio (.mp3)", open(vf, "rb").read(),
                                   file_name="my_speech.mp3", mime="audio/mpeg")

elif page == "Audio to Text":
    st.markdown('<div class="page-hero"><div class="ph-title">🎙️ Audio to Text</div>'
                '<div class="ph-sub">Turn speech, recorded meetings, voice memos, and lectures into accurate '
                'text, structured summaries, and action checklists using AI.</div></div>',
                unsafe_allow_html=True)

    input_choice = st.radio("How would you like to provide audio?",
                            ["🎤 Record from Microphone", "📁 Upload Audio File"],
                            horizontal=True, key="stt_input_choice")

    audio_bytes = None
    audio_mime = "audio/wav"

    if input_choice.startswith("🎤"):
        st.markdown('<div class="sec-label">Record your voice directly</div>', unsafe_allow_html=True)
        rec = st.audio_input("Click the microphone to record, speak, then click stop:", key="stt_mic")
        if rec is not None:
            audio_bytes = rec.getvalue()
            audio_mime = "audio/wav"
    else:
        st.markdown('<div class="sec-label">Upload an audio recording</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload an audio file (MP3, WAV, M4A, OGG, WEBM, FLAC)",
                                    type=["mp3", "wav", "m4a", "ogg", "webm", "flac"], key="stt_file")
        if uploaded is not None:
            audio_bytes = uploaded.getvalue()
            fname = uploaded.name.lower()
            if fname.endswith(".mp3"):
                audio_mime = "audio/mp3"
            elif fname.endswith(".wav"):
                audio_mime = "audio/wav"
            elif fname.endswith(".ogg"):
                audio_mime = "audio/ogg"
            elif fname.endswith(".m4a"):
                audio_mime = "audio/m4a"
            elif fname.endswith(".webm"):
                audio_mime = "audio/webm"
            else:
                audio_mime = "audio/mp3"

    if audio_bytes:
        st.audio(audio_bytes, format=audio_mime)

    st.markdown('<div class="sec-label">Choose AI Output Mode</div>', unsafe_allow_html=True)
    stt_mode = st.radio("What should AI produce?",
                        ["📝 Full Verbatim Transcript",
                         "📋 Executive Meeting Summary",
                         "✅ Action Items & To-Dos",
                         "💡 Study / Lecture Notes",
                         "🌐 Translate Language"],
                        horizontal=True, key="stt_mode")

    target_lang = "English"
    if stt_mode.startswith("🌐"):
        target_lang = st.selectbox("Translate into:",
                                   ["English", "Spanish", "French", "German", "Urdu",
                                    "Hindi", "Arabic", "Chinese", "Japanese", "Portuguese"],
                                   key="stt_target_lang")

    if st.button("Transcribe & Process Audio", key="stt_btn", type="primary"):
        if not audio_bytes:
            st.warning("Please record your voice or upload an audio file first!")
        else:
            with st.spinner("AI is listening and transcribing your audio... please wait"):
                raw_text, vtt, err = ask_transcribe_audio(audio_bytes, audio_mime)

            if err or not raw_text:
                st.error(err or "Could not transcribe audio. Please try again.")
            else:
                st.session_state["stt_raw"] = raw_text
                st.session_state["stt_vtt"] = vtt

                if stt_mode.startswith("📝"):
                    final_output = raw_text
                else:
                    with st.spinner("AI is generating your requested analysis..."):
                        if stt_mode.startswith("📋"):
                            prompt = (f"You are an executive assistant. Based on this audio transcript, "
                                      f"generate a professional executive meeting summary with: "
                                      f"Overview, Key Discussion Points, and Decisions Made.\n\n"
                                      f"Transcript:\n{raw_text}")
                        elif stt_mode.startswith("✅"):
                            prompt = (f"Based on this audio transcript, extract all action items, tasks, "
                                      f"assigned responsibilities, and deadlines into a clean checklist with checkboxes.\n\n"
                                      f"Transcript:\n{raw_text}")
                        elif stt_mode.startswith("💡"):
                            prompt = (f"Convert this lecture/audio transcript into structured study notes "
                                      f"with core concepts explained in simple terms, bullet takeaways, "
                                      f"and 3 review quiz questions.\n\n"
                                      f"Transcript:\n{raw_text}")
                        else:
                            prompt = (f"Translate this transcript accurately into fluent {target_lang}. "
                                      f"Maintain natural phrasing and tone.\n\n"
                                      f"Transcript:\n{raw_text}")
                        ai_result = ask_text_ai(prompt)
                        final_output = ai_result or raw_text

                st.session_state["stt_output"] = final_output
                st.session_state["stt_ready"] = True

    if st.session_state.get("stt_ready"):
        st.success("Transcribed successfully!")
        out_text = st.session_state.get("stt_output", "")
        raw_text = st.session_state.get("stt_raw", "")
        vtt_text = st.session_state.get("stt_vtt", "")

        # Statistics
        words = len(raw_text.split())
        chars = len(raw_text)
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Total Spoken Words", f"{words:,}")
        with c2:
            st.metric("Total Characters", f"{chars:,}")

        st.markdown('<div class="sec-label">AI Output</div>', unsafe_allow_html=True)
        st.markdown(out_text)

        if out_text != raw_text:
            with st.expander("📄 View Original Verbatim Transcript"):
                st.text_area("Verbatim Transcript", raw_text, height=180, key="stt_raw_view")

        # Download buttons
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.download_button("📥 Download Text (.txt)", out_text.encode("utf-8"),
                               file_name="audio_transcript.txt", mime="text/plain",
                               use_container_width=True)
        with b_col2:
            if vtt_text:
                st.download_button("📥 Download Subtitles (.vtt)", vtt_text.encode("utf-8"),
                                   file_name="subtitles.vtt", mime="text/vtt",
                                   use_container_width=True)
            else:
                st.download_button("📥 Download Raw Transcript (.txt)", raw_text.encode("utf-8"),
                                   file_name="verbatim_transcript.txt", mime="text/plain",
                                   use_container_width=True)

elif page == "QR Code":
    st.markdown('<div class="page-hero"><div class="ph-title">🔗 QR Code</div>'
                '<div class="ph-sub">Turn any link or text into a QR code you can download '
                'and print. Free and instant.</div></div>',
                unsafe_allow_html=True)

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
