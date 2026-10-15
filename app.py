import os
import re
import io
import json
import time
import random
import requests
import streamlit as st
from fpdf import FPDF
from PIL import Image, ImageOps, ImageDraw, ImageFont

# Optional document parsing libraries
try:
    import pypdf
except ImportError:
    pypdf = None

try:
    import docx
except ImportError:
    docx = None

# Optional PDF-to-image renderer for real template previews
try:
    import pypdfium2 as pdfium
except ImportError:
    pdfium = None

# ---- 0. Load Environment Variables (.env locally, st.secrets on Cloud) ----
def load_env():
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    try:
        for key in st.secrets:
            os.environ.setdefault(key, str(st.secrets[key]))
    except Exception:
        pass

load_env()

# ---- 1. AI Text Engine for Resume Writing, Parsing & ATS Optimization ----
BUSY_MSG = "Sorry, the AI is busy right now. Please try again in a moment."

def ask_gemini_text(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "no GEMINI_API_KEY in .env"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    try:
        r = requests.post(url, params={"key": api_key},
                          json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=90)
        if r.status_code != 200:
            return None, f"Gemini error {r.status_code}: {r.text[:200]}"
        parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts), None
    except Exception as e:
        return None, str(e)

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
    # 1. Primary: Groq (ultra-fast Llama 3.3 70B)
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
            data = {"model": "llama-3.3-70b-versatile", "messages": messages}
            r = requests.post(url, headers=headers, json=data, timeout=40)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"]
        except Exception:
            pass

    # 2. Secondary: Gemini 2.5 Flash
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

    # 3. Tertiary: Cloudflare Workers AI
    cf_res = ask_cf_text(combined_prompt)
    if cf_res:
        return cf_res
    return None

# ---- 2. Old Resume Text Extraction & AI Parsing ----
def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    content = uploaded_file.getvalue()

    if name.endswith(".pdf"):
        if not pypdf:
            return None, "pypdf library is not installed."
        try:
            reader = pypdf.PdfReader(io.BytesIO(content))
            texts = [page.extract_text() or "" for page in reader.pages]
            full_text = "\n".join(texts).strip()
            if not full_text:
                return None, "Could not extract readable text from this PDF (it might be a scanned image)."
            return full_text, None
        except Exception as e:
            return None, f"Error reading PDF: {e}"

    elif name.endswith(".docx"):
        if not docx:
            return None, "python-docx library is not installed."
        try:
            doc = docx.Document(io.BytesIO(content))
            full_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
            return full_text, None
        except Exception as e:
            return None, f"Error reading DOCX: {e}"

    else:
        try:
            return content.decode("utf-8", errors="replace"), None
        except Exception as e:
            return None, f"Error reading text file: {e}"

def parse_resume_with_ai(raw_text):
    prompt = (
        f"You are a Fortune 500 professional resume parsing AI.\n"
        f"Extract all candidate profile information from the following resume text into clean, structured JSON.\n\n"
        f"Resume Content:\n{raw_text[:7500]}\n\n"
        f"You MUST reply ONLY with a valid JSON object matching this schema (if a field is missing, set it to empty string):\n"
        f"```json\n"
        f"{{\n"
        f'  "name": "Candidate Full Name",\n'
        f'  "title": "Current or Target Job Title",\n'
        f'  "email": "Email Address",\n'
        f'  "phone": "Phone Number",\n'
        f'  "location": "City, State / Country",\n'
        f'  "links": "LinkedIn / GitHub / Portfolio URLs",\n'
        f'  "skills": "Comma-separated list of technical & core skills",\n'
        f'  "experience": "Detailed work experience with role, company, duration, and bulleted achievements",\n'
        f'  "projects": "Key technical projects with technologies and impact",\n'
        f'  "education": "Degrees, university, graduation year"\n'
        f"}}\n"
        f"```\n"
        f"Do not include any conversational preamble or notes. Output ONLY the JSON."
    )
    raw_ai = ask_text_ai(prompt)
    if not raw_ai:
        return None
    try:
        match = re.search(r"\{[\s\S]*\}", raw_ai)
        if match:
            return json.loads(match.group(0))
        return json.loads(raw_ai)
    except Exception:
        return None

# ---- 3. Profile Photo Processing ----
def prepare_avatar(img_bytes, size=(300, 300)):
    try:
        im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        im = ImageOps.fit(im, size, centering=(0.5, 0.5))
        mask = Image.new("L", size, 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size[0], size[1]), fill=255)
        output = Image.new("RGBA", size, (255, 255, 255, 0))
        output.paste(im, (0, 0), mask=mask)
        final = Image.new("RGB", size, (255, 255, 255))
        final.paste(output, mask=output.split()[3])
        buf = io.BytesIO()
        final.save(buf, format="JPEG", quality=95)
        return buf.getvalue()
    except Exception:
        return None

# ---- 4. Generate Template Preview Thumbnails (Programmatic) ----
def generate_template_thumbnail(tmpl):
    """Generate a realistic resume preview thumbnail image for a template."""
    w, h = 280, 396  # A4 ratio thumbnail
    accent = tmpl.get("accent_rgb", (30, 58, 138))
    accent2 = tmpl.get("accent2_rgb", (59, 130, 246))
    layout = tmpl.get("layout", "single")

    img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    if "sidebar" in layout.lower():
        # Sidebar layout
        sb_w = 90
        draw.rectangle([0, 0, sb_w, h], fill=(*accent, ))
        # Photo circle placeholder
        cx, cy, r = sb_w // 2, 50, 22
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(255, 255, 255))
        # Sidebar lines
        for i in range(8):
            y = 90 + i * 18
            lw = random.randint(35, 70)
            draw.rectangle([12, y, 12 + lw, y + 6], fill=(255, 255, 255, 180))
        # Right side content
        draw.rectangle([sb_w + 15, 20, sb_w + 15 + 140, 28], fill=accent)
        draw.rectangle([sb_w + 15, 34, sb_w + 15 + 90, 40], fill=(*accent2,))
        for i in range(12):
            y = 60 + i * 22
            lw = random.randint(80, 160)
            draw.rectangle([sb_w + 15, y, sb_w + 15 + lw, y + 5], fill=(200, 200, 210))
            if i % 3 == 0:
                draw.rectangle([sb_w + 15, y, sb_w + 15 + lw, y + 5], fill=(*accent, 80))

    elif "banner" in layout.lower():
        # Top banner layout
        draw.rectangle([0, 0, w, 60], fill=accent)
        draw.rectangle([20, 16, 160, 26], fill=(255, 255, 255))
        draw.rectangle([20, 32, 120, 38], fill=(200, 210, 240))
        # Photo circle in banner
        draw.ellipse([w - 55, 10, w - 15, 50], fill=(255, 255, 255))
        # Body content
        for i in range(14):
            y = 75 + i * 20
            lw = random.randint(100, 240)
            color = accent if i % 5 == 0 else (200, 200, 210)
            draw.rectangle([20, y, 20 + lw, y + 5], fill=color)

    else:
        # Clean single column
        draw.rectangle([0, 0, w, 5], fill=accent)
        # Name
        draw.rectangle([60, 18, 220, 30], fill=accent)
        draw.rectangle([80, 36, 200, 42], fill=(*accent2,))
        # Contact line
        draw.rectangle([50, 48, 230, 53], fill=(180, 180, 190))
        draw.rectangle([20, 58, 260, 59], fill=(220, 225, 235))
        # Sections
        for i in range(13):
            y = 70 + i * 22
            if i % 4 == 0:
                draw.rectangle([20, y, 20 + 80, y + 7], fill=accent)
                draw.rectangle([20, y + 9, 260, y + 10], fill=(220, 225, 235))
            else:
                lw = random.randint(100, 240)
                draw.rectangle([28, y, 28 + lw, y + 5], fill=(200, 200, 210))

    # Add subtle border
    draw.rectangle([0, 0, w - 1, h - 1], outline=(220, 220, 230), width=1)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---- 5. Template & Theme Configurations ----
RESUME_THEMES = {
    "Modern Navy": {
        "primary": (30, 58, 138),
        "accent": (59, 130, 246),
        "sidebar_bg": (244, 246, 250),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Emerald Professional": {
        "primary": (6, 95, 70),
        "accent": (16, 185, 129),
        "sidebar_bg": (242, 248, 245),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Crimson Executive": {
        "primary": (159, 18, 57),
        "accent": (225, 29, 72),
        "sidebar_bg": (254, 242, 244),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Royal Purple": {
        "primary": (88, 28, 135),
        "accent": (147, 51, 234),
        "sidebar_bg": (248, 244, 253),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Classic Charcoal": {
        "primary": (31, 41, 55),
        "accent": (75, 85, 99),
        "sidebar_bg": (245, 246, 248),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Ocean Teal": {
        "primary": (13, 148, 136),
        "accent": (45, 212, 191),
        "sidebar_bg": (240, 253, 250),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Sunset Orange": {
        "primary": (194, 65, 12),
        "accent": (251, 146, 60),
        "sidebar_bg": (255, 247, 237),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
    "Slate Blue": {
        "primary": (51, 65, 85),
        "accent": (100, 116, 139),
        "sidebar_bg": (248, 250, 252),
        "text": (17, 24, 39),
        "muted": (75, 85, 99),
        "line": (226, 232, 240),
    },
}

# Job categories for sidebar filter
CATEGORIES = [
    "All Categories", "AI & Machine Learning", "Software Engineering", "Data Science",
    "Web Development", "Mobile Development", "Cloud & DevOps", "Cybersecurity",
    "Business Analyst", "Product Manager", "UI/UX Design", "Digital Marketing",
    "Finance & Accounting", "Human Resources", "Sales & Marketing", "Healthcare",
    "Education & Teaching", "Engineering", "Consulting", "Legal",
    "Content Writing", "Graphic Design", "Project Management", "Operations",
]

TEMPLATES = [
    {
        "id": "modern_clean",
        "name": "Modern Clean Resume",
        "category": "All Categories",
        "badge": "🏆 #1 ATS Pick",
        "desc": "Single-column executive layout favored by Fortune 500 recruiters and ATS parsers.",
        "theme": "Modern Navy",
        "layout": "Modern Clean (Single Column)",
        "accent_color": "#1e3a8a",
        "accent_rgb": (30, 58, 138),
        "accent2_rgb": (59, 130, 246),
        "likes": 66751,
        "tags": ["ATS-Friendly", "Professional", "Clean"],
    },
    {
        "id": "modern_sidebar",
        "name": "Executive Dual-Tone",
        "category": "All Categories",
        "badge": "🎨 Visual Balance",
        "desc": "Stylish two-column format with a tinted sidebar for skills, contact & degrees.",
        "theme": "Royal Purple",
        "layout": "Modern Sidebar (Dual Tone)",
        "accent_color": "#581c87",
        "accent_rgb": (88, 28, 135),
        "accent2_rgb": (147, 51, 234),
        "likes": 45230,
        "tags": ["Creative", "Dual-Column", "Design"],
    },
    {
        "id": "emerald_tech",
        "name": "Emerald Tech Lead",
        "category": "Software Engineering",
        "badge": "💻 Tech & Data",
        "desc": "Emerald-accented engineering format emphasizing technical architecture and metrics.",
        "theme": "Emerald Professional",
        "layout": "Modern Clean (Single Column)",
        "accent_color": "#065f46",
        "accent_rgb": (6, 95, 70),
        "accent2_rgb": (16, 185, 129),
        "likes": 32890,
        "tags": ["Engineering", "Technical", "Clean"],
    },
    {
        "id": "crimson_director",
        "name": "Crimson Director",
        "category": "Finance & Accounting",
        "badge": "👔 Finance & Strategy",
        "desc": "Prestigious burgundy styling tailored for Directors, Consultants, and Finance leads.",
        "theme": "Crimson Executive",
        "layout": "Modern Sidebar (Dual Tone)",
        "accent_color": "#9f1239",
        "accent_rgb": (159, 18, 57),
        "accent2_rgb": (225, 29, 72),
        "likes": 28450,
        "tags": ["Executive", "Finance", "Premium"],
    },
    {
        "id": "executive_banner",
        "name": "Corporate Header Banner",
        "category": "Business Analyst",
        "badge": "🌟 Premium Banner",
        "desc": "Distinct colored accent banner with prominent name branding and streamlined layout.",
        "theme": "Modern Navy",
        "layout": "Executive Banner (Top Accent)",
        "accent_color": "#1d4ed8",
        "accent_rgb": (30, 78, 216),
        "accent2_rgb": (96, 165, 250),
        "likes": 20113,
        "tags": ["Corporate", "Banner", "Executive"],
    },
    {
        "id": "nordic_minimal",
        "name": "Nordic Minimalist",
        "category": "UI/UX Design",
        "badge": "📄 Elegant",
        "desc": "Sleek monochromatic structure with generous breathing room and clean typography.",
        "theme": "Classic Charcoal",
        "layout": "Modern Clean (Single Column)",
        "accent_color": "#1f2937",
        "accent_rgb": (31, 41, 55),
        "accent2_rgb": (75, 85, 99),
        "likes": 18340,
        "tags": ["Minimal", "Clean", "Whitespace"],
    },
    {
        "id": "ocean_creative",
        "name": "Ocean Creative Resume",
        "category": "Digital Marketing",
        "badge": "🌊 Creative",
        "desc": "Teal-accented modern design perfect for creative professionals and marketing roles.",
        "theme": "Ocean Teal",
        "layout": "Modern Sidebar (Dual Tone)",
        "accent_color": "#0d9488",
        "accent_rgb": (13, 148, 136),
        "accent2_rgb": (45, 212, 191),
        "likes": 15620,
        "tags": ["Creative", "Marketing", "Teal"],
    },
    {
        "id": "sunset_manager",
        "name": "Sunset Project Manager",
        "category": "Project Management",
        "badge": "🔥 Bold Impact",
        "desc": "Warm orange tones with strong visual hierarchy for project managers and leaders.",
        "theme": "Sunset Orange",
        "layout": "Executive Banner (Top Accent)",
        "accent_color": "#c2410c",
        "accent_rgb": (194, 65, 12),
        "accent2_rgb": (251, 146, 60),
        "likes": 12890,
        "tags": ["Management", "Bold", "Warm"],
    },
    {
        "id": "ai_specialist",
        "name": "AI & ML Specialist",
        "category": "AI & Machine Learning",
        "badge": "🤖 AI Ready",
        "desc": "Designed for AI engineers, data scientists, and ML researchers with a tech-forward look.",
        "theme": "Modern Navy",
        "layout": "Modern Clean (Single Column)",
        "accent_color": "#1e3a8a",
        "accent_rgb": (30, 58, 138),
        "accent2_rgb": (99, 102, 241),
        "likes": 24560,
        "tags": ["AI", "Machine Learning", "Technical"],
    },
    {
        "id": "slate_professional",
        "name": "Slate Professional",
        "category": "Consulting",
        "badge": "🏛️ Corporate",
        "desc": "Understated slate-blue corporate design for consultants and business professionals.",
        "theme": "Slate Blue",
        "layout": "Modern Clean (Single Column)",
        "accent_color": "#334155",
        "accent_rgb": (51, 65, 85),
        "accent2_rgb": (100, 116, 139),
        "likes": 11230,
        "tags": ["Corporate", "Consulting", "Subtle"],
    },
    {
        "id": "purple_creative",
        "name": "Purple Creative Portfolio",
        "category": "Graphic Design",
        "badge": "🎭 Portfolio",
        "desc": "Vibrant purple dual-column layout for designers, artists, and creative professionals.",
        "theme": "Royal Purple",
        "layout": "Modern Sidebar (Dual Tone)",
        "accent_color": "#7c3aed",
        "accent_rgb": (124, 58, 237),
        "accent2_rgb": (167, 139, 250),
        "likes": 19870,
        "tags": ["Design", "Creative", "Portfolio"],
    },
    {
        "id": "healthcare_green",
        "name": "Healthcare Professional",
        "category": "Healthcare",
        "badge": "🏥 Medical",
        "desc": "Clean emerald design for doctors, nurses, and healthcare workers with clear sections.",
        "theme": "Emerald Professional",
        "layout": "Executive Banner (Top Accent)",
        "accent_color": "#065f46",
        "accent_rgb": (6, 95, 70),
        "accent2_rgb": (52, 211, 153),
        "likes": 14350,
        "tags": ["Healthcare", "Medical", "Professional"],
    },
]

# ---- 6. Executive PDF Engine ----
def make_resume_pdf(text, header=None, theme="Modern Navy", layout="Modern Clean", photo_bytes=None):
    theme_info = RESUME_THEMES.get(theme, RESUME_THEMES["Modern Navy"])
    pdf = FPDF("p", "mm", "A4")
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

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

    replacements = {
        "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ",
        "\u2011": "-", "\u2013": "-", "\u2014": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for smart, normal in replacements.items():
        text = text.replace(smart, normal)

    if font == "Helvetica":
        text = text.encode("latin-1", errors="replace").decode("latin-1")

    text = re.sub(r"\[[^\]]{0,60}\]", "", text)

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
    avatar_data = prepare_avatar(photo_bytes) if photo_bytes else None

    is_sidebar = "sidebar" in layout.lower()
    is_banner = "banner" in layout.lower()

    if is_sidebar:
        sb_w = 66
        main_x = sb_w + 10
        main_w = 210 - main_x - 14

        pdf.set_fill_color(*theme_info["sidebar_bg"])
        pdf.rect(0, 0, sb_w, 297, style="F")
        pdf.set_fill_color(*theme_info["primary"])
        pdf.rect(sb_w, 0, 210 - sb_w, 3.5, style="F")

        SB_KEYS = ("contact", "skill", "educat", "personal", "detail", "certif", "language", "hobby")
        sb_secs = [s for s in sections if any(k in s["title"].lower() for k in SB_KEYS)]
        main_secs = [s for s in sections if not any(k in s["title"].lower() for k in SB_KEYS)]

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

        sb_start_y = 14
        if avatar_data:
            avatar_w = 30
            avatar_x = (sb_w - avatar_w) / 2
            pdf.image(io.BytesIO(avatar_data), x=avatar_x, y=14, w=avatar_w, h=avatar_w)
            sb_start_y = 14 + avatar_w + 6

        pdf.set_xy(10, sb_start_y)
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

    elif is_banner:
        MARGIN = 16
        col_w = 210 - 2 * MARGIN
        pdf.set_fill_color(*theme_info["primary"])
        banner_h = 36 if avatar_data else 32
        pdf.rect(0, 0, 210, banner_h, style="F")
        text_x = MARGIN
        text_w = col_w
        if avatar_data:
            pdf.image(io.BytesIO(avatar_data), x=MARGIN, y=4, w=24, h=24)
            text_x = MARGIN + 28
            text_w = col_w - 28
        pdf.set_xy(text_x, 6)
        pdf.set_font(font, "B", 20)
        pdf.set_text_color(255, 255, 255)
        pdf.cell(text_w, 8, header.get("name", "Candidate Name"), new_x="LMARGIN", new_y="NEXT")
        if header.get("title"):
            pdf.set_x(text_x)
            pdf.set_font(font, "B", 10.5)
            pdf.set_text_color(224, 231, 255)
            pdf.cell(text_w, 5.5, header["title"].upper(), new_x="LMARGIN", new_y="NEXT")
        if header.get("contact"):
            pdf.set_x(text_x)
            pdf.set_font(font, "", 8.8)
            pdf.set_text_color(243, 244, 246)
            pdf.cell(text_w, 5, header["contact"], new_x="LMARGIN", new_y="NEXT")
        pdf.set_y(banner_h + 6)
        total_lines = sum(len(s["body"]) for s in sections)
        LH = 5.2 if total_lines < 32 else 4.8
        sec_gap = 4.2 if total_lines < 32 else 2.6
        for sec in sections:
            y = pdf.get_y()
            if y > 275:
                pdf.add_page()
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

    else:
        MARGIN = 16
        col_w = 210 - 2 * MARGIN
        pdf.set_fill_color(*theme_info["primary"])
        pdf.rect(0, 0, 210, 3.5, style="F")
        if avatar_data:
            pdf.image(io.BytesIO(avatar_data), x=MARGIN, y=10, w=24, h=24)
            h_x = MARGIN + 28
            h_w = col_w - 28
            pdf.set_xy(h_x, 10)
            pdf.set_font(font, "B", 20)
            pdf.set_text_color(*theme_info["text"])
            pdf.cell(h_w, 8, header.get("name", "Candidate Name"), new_x="LMARGIN", new_y="NEXT")
            if header.get("title"):
                pdf.set_x(h_x)
                pdf.set_font(font, "B", 10.5)
                pdf.set_text_color(*theme_info["primary"])
                pdf.cell(h_w, 5.5, header["title"].upper(), new_x="LMARGIN", new_y="NEXT")
            if header.get("contact"):
                pdf.set_x(h_x)
                pdf.set_font(font, "", 9.0)
                pdf.set_text_color(*theme_info["muted"])
                pdf.cell(h_w, 5, header["contact"], new_x="LMARGIN", new_y="NEXT")
            pdf.set_y(38)
        else:
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
            pdf.set_y(pdf.get_y() + 1)
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


# ============================================================
# ---- 7. APP UI SETUP & PROFESSIONAL STYLES ----
# ============================================================
st.set_page_config(
    page_title="ResumeStudio — Professional Resume Templates",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 1rem; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label { color: #374151; }

    .sidebar-brand {
        font-size: 24px; font-weight: 800; color: #1e40af;
        padding: 8px 0 2px; letter-spacing: -0.5px;
    }
    .sidebar-tagline {
        font-size: 12px; color: #6b7280;
        margin-bottom: 16px; line-height: 1.4;
    }
    .cat-header {
        font-size: 11px; font-weight: 700; color: #9ca3af;
        text-transform: uppercase; letter-spacing: 1.5px;
        margin: 18px 0 8px; padding-bottom: 6px;
        border-bottom: 1px solid #f3f4f6;
    }

    /* ── Hero Banner ── */
    .hero-banner {
        background: linear-gradient(135deg, #1e3a5f 0%, #1e40af 40%, #3b82f6 100%);
        border-radius: 16px; padding: 32px 40px 26px;
        margin-bottom: 28px; color: #ffffff;
        box-shadow: 0 8px 30px rgba(30, 64, 175, 0.2);
    }
    .hero-banner h1 {
        font-size: 36px; font-weight: 900; margin: 0 0 6px;
        letter-spacing: -0.5px; line-height: 1.1;
    }
    .hero-banner p {
        font-size: 14.5px; opacity: 0.92; margin: 0; line-height: 1.5;
    }
    .hero-stats-bar {
        display: flex; gap: 32px; margin-top: 18px;
        flex-wrap: wrap;
    }
    .hero-stat { text-align: center; }
    .hero-stat .num { font-size: 22px; font-weight: 800; display: block; }
    .hero-stat .lbl {
        font-size: 10px; text-transform: uppercase;
        letter-spacing: 1px; opacity: 0.8;
    }

    /* ── Filter Tabs ── */
    .filter-bar {
        display: flex; align-items: center; gap: 0;
        margin-bottom: 20px;
    }
    .filter-tab {
        padding: 8px 24px; font-size: 14px; font-weight: 600;
        border: 2px solid #e5e7eb; cursor: pointer;
        background: #ffffff; color: #6b7280;
        transition: all 0.2s;
    }
    .filter-tab:first-child { border-radius: 8px 0 0 8px; }
    .filter-tab:last-child { border-radius: 0 8px 8px 0; }
    .filter-tab.active {
        background: #1e40af; color: #ffffff;
        border-color: #1e40af;
    }

    /* ── Template Cards ── */
    .tmpl-card {
        background: #ffffff; border: 1px solid #e5e7eb;
        border-radius: 12px; overflow: hidden;
        transition: all 0.25s ease;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .tmpl-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 28px rgba(0,0,0,0.12);
        border-color: #3b82f6;
    }
    .tmpl-img-wrap {
        position: relative; background: #f9fafb;
        padding: 12px; display: flex; justify-content: center;
        min-height: 220px; align-items: center;
    }
    .tmpl-img-wrap img {
        max-height: 200px; border-radius: 4px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .tmpl-overlay {
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(30, 64, 175, 0.7);
        display: flex; align-items: center; justify-content: center;
        opacity: 0; transition: opacity 0.25s; border-radius: 0;
    }
    .tmpl-card:hover .tmpl-overlay { opacity: 1; }
    .tmpl-overlay-btn {
        background: #ffffff; color: #1e40af;
        padding: 10px 28px; border-radius: 8px;
        font-weight: 700; font-size: 14px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    .tmpl-info { padding: 14px 16px 16px; }
    .tmpl-info .tmpl-title {
        font-size: 15px; font-weight: 700; color: #111827;
        margin-bottom: 4px; line-height: 1.3;
    }
    .tmpl-info .tmpl-tags {
        font-size: 11px; color: #9ca3af; margin-bottom: 6px;
    }
    .tmpl-info .tmpl-likes {
        font-size: 13px; color: #ef4444; font-weight: 600;
    }
    .tmpl-info .tmpl-likes span { color: #6b7280; font-weight: 400; }

    /* ── Section Headings ── */
    .section-head {
        font-size: 24px; font-weight: 800; color: #111827;
        margin: 6px 0 4px; letter-spacing: -0.3px;
    }
    .section-sub {
        font-size: 13.5px; color: #6b7280;
        margin-bottom: 18px; line-height: 1.4;
    }

    /* ── Buttons ── */
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stDownloadButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #1e40af, #3b82f6);
        color: #ffffff; border: none; border-radius: 8px;
        font-weight: 700; padding: 0.5rem 1.2rem;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover,
    div[data-testid="stDownloadButton"] button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(59, 130, 246, 0.35);
        color: #ffffff;
    }

    /* ── Builder page ── */
    .builder-header {
        background: linear-gradient(135deg, #1e3a5f, #1e40af);
        border-radius: 12px; padding: 24px 30px; color: #fff;
        margin-bottom: 20px;
    }
    .builder-header h2 { margin: 0 0 4px; font-weight: 800; }
    .builder-header p { margin: 0; opacity: 0.85; font-size: 14px; }

    /* Chat Input */
    div[data-testid="stChatInput"] {
        border: 2px solid #93c5fd; border-radius: 12px;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ---- 8. Session State Initialization ----
defaults = {
    "page": "gallery",  # "gallery" or "builder"
    "resume_text": "",
    "resume_log": [],
    "resume_header": {},
    "chosen_template_id": None,
    "photo_bytes": None,
    "include_photo": True,
    "selected_category": "All Categories",
    "sort_mode": "Popular",
    # Input Form Fields
    "input_name": "",
    "input_title": "",
    "input_email": "",
    "input_phone": "",
    "input_address": "",
    "input_links": "",
    "input_skills": "",
    "input_experience": "",
    "input_projects": "",
    "input_education": "",
    "input_father": "",
    "input_dob": "",
    "input_gender": "Not specified",
    "input_marital": "Not specified",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---- Sample resume used only to render the real template previews ----
SAMPLE_HEADER = {
    "name": "Alex Morgan",
    "title": "Senior Software Engineer",
    "email": "alex.morgan@email.com",
    "phone": "+1 (555) 123-4567",
    "contact": "San Francisco, CA   |   alex.morgan@email.com   |   +1 (555) 123-4567   |   linkedin.com/in/alexmorgan",
}

SAMPLE_RESUME_TEXT = """## Summary
Results-driven Senior Software Engineer with 8+ years building scalable web platforms and data-driven services. Led cross-functional teams to ship products used by millions.

## Experience
### Senior Software Engineer - TechNova Inc. (2021 - Present)
- Architected a microservices platform that cut deploy time by 60% and improved uptime to 99.98%.
- Mentored a team of 5 engineers and established code-review and testing standards.
- Reduced cloud infrastructure cost by 35% through autoscaling and right-sizing.
### Software Engineer - BrightApps (2017 - 2021)
- Built REST APIs and real-time dashboards serving 2M+ monthly active users.
- Automated CI/CD pipelines, shrinking release cycles from weekly to daily.

## Skills
- Languages: Python, JavaScript, TypeScript, Go, SQL
- Frameworks: React, Node.js, Django, FastAPI
- Cloud & Tools: AWS, Docker, Kubernetes, PostgreSQL, Redis, Git

## Projects
- Open-source analytics toolkit with 3,000+ GitHub stars.
- Real-time recommendation engine boosting engagement by 22%.

## Education
### B.Tech, Computer Science - State University (2013 - 2017)
Graduated with Honors.
"""


def render_template_preview(tmpl):
    """Render a real one-page PDF of the sample resume in this template's theme/layout,
    then rasterize page 1 to a PNG. Falls back to the drawn mock thumbnail on any failure."""
    try:
        if pdfium is None:
            raise RuntimeError("pypdfium2 not installed")
        pdf_bytes = make_resume_pdf(
            SAMPLE_RESUME_TEXT, SAMPLE_HEADER,
            theme=tmpl.get("theme"), layout=tmpl.get("layout"), photo_bytes=None,
        )
        doc = pdfium.PdfDocument(pdf_bytes)
        pil = doc[0].render(scale=1.4).to_pil().convert("RGB")
        target_w = 300
        target_h = int(pil.height * (target_w / pil.width))
        pil = pil.resize((target_w, target_h), Image.LANCZOS)
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        return buf.getvalue()
    except Exception:
        return generate_template_thumbnail(tmpl)


# Pre-generate thumbnails (cached)
@st.cache_data(show_spinner="Rendering template previews...")
def get_all_thumbnails():
    thumbs = {}
    for tmpl in TEMPLATES:
        thumbs[tmpl["id"]] = render_template_preview(tmpl)
    return thumbs

THUMBNAILS = get_all_thumbnails()

# ============================================================
# ---- 9. SIDEBAR: Categories & Branding ----
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📄 ResumeStudio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Professional Resume Templates<br>Free Builder • ATS-Optimized • PDF Export</div>', unsafe_allow_html=True)

    # Back to gallery button (when in builder)
    if st.session_state["page"] == "builder":
        if st.button("← Back to Templates", use_container_width=True):
            st.session_state["page"] = "gallery"
            st.rerun()
        st.markdown("---")

    st.markdown('<div class="cat-header">CATEGORIES</div>', unsafe_allow_html=True)

    for cat in CATEGORIES:
        is_active = st.session_state["selected_category"] == cat
        label = f"**{cat}**" if is_active else cat
        if st.button(label, key=f"cat_{cat}", use_container_width=True):
            st.session_state["selected_category"] = cat
            st.session_state["page"] = "gallery"
            st.rerun()

    st.markdown("---")
    groq_ok = bool(os.environ.get("GROQ_API_KEY"))
    gemini_ok = bool(os.environ.get("GEMINI_API_KEY"))
    if groq_ok or gemini_ok:
        st.caption("🟢 AI Engine: Connected")
    else:
        st.caption("⚠️ Add API keys in `.env` for AI features")


# ============================================================
# ---- 10. PAGE: TEMPLATE GALLERY (Landing Page) ----
# ============================================================
if st.session_state["page"] == "gallery":

    # Hero Banner
    cat = st.session_state["selected_category"]
    cat_display = cat if cat != "All Categories" else "All"
    st.markdown(f"""
    <div class="hero-banner">
        <h1>Resume Templates on ResumeStudio</h1>
        <p>All resumes are created using ResumeStudio's free AI Resume Builder — pick a template, fill your details, download your PDF.</p>
        <div class="hero-stats-bar">
            <div class="hero-stat"><span class="num">{len(TEMPLATES)}</span><span class="lbl">Templates</span></div>
            <div class="hero-stat"><span class="num">📸</span><span class="lbl">Photo Support</span></div>
            <div class="hero-stat"><span class="num">📤</span><span class="lbl">Resume Import</span></div>
            <div class="hero-stat"><span class="num">ATS 95+</span><span class="lbl">Optimized</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Popular / Latest tabs
    col_f1, col_f2, _ = st.columns([1, 1, 4])
    with col_f1:
        if st.button("🔥 Popular", type="primary" if st.session_state["sort_mode"] == "Popular" else "secondary",
                      use_container_width=True, key="btn_popular"):
            st.session_state["sort_mode"] = "Popular"
            st.rerun()
    with col_f2:
        if st.button("🆕 Latest", type="primary" if st.session_state["sort_mode"] == "Latest" else "secondary",
                      use_container_width=True, key="btn_latest"):
            st.session_state["sort_mode"] = "Latest"
            st.rerun()

    # Filter templates by category
    selected_cat = st.session_state["selected_category"]
    if selected_cat == "All Categories":
        filtered = TEMPLATES[:]
    else:
        filtered = [t for t in TEMPLATES if t["category"] == selected_cat]
        # Also show "All Categories" templates
        for t in TEMPLATES:
            if t["category"] == "All Categories" and t not in filtered:
                filtered.append(t)

    # Sort
    if st.session_state["sort_mode"] == "Popular":
        filtered.sort(key=lambda t: t["likes"], reverse=True)
    else:
        filtered.sort(key=lambda t: t["likes"])  # "Latest" = reverse of popular for demo

    if not filtered:
        st.info(f"No templates found for **{selected_cat}**. Try 'All Categories'.")
    else:
        # Display in 3-column grid
        cols = st.columns(3)
        for i, tmpl in enumerate(filtered):
            with cols[i % 3]:
                # Template preview image
                thumb_bytes = THUMBNAILS.get(tmpl["id"])
                if thumb_bytes:
                    st.image(thumb_bytes, use_container_width=True)

                # Template info below image
                likes_str = f"{tmpl['likes']:,}"
                tags_str = " • ".join(tmpl.get("tags", []))
                st.markdown(f"""
                <div style="padding: 4px 0 12px;">
                    <div style="font-size:15px; font-weight:700; color:#111827; margin-bottom:2px;">{tmpl['name']}</div>
                    <div style="font-size:11.5px; color:#9ca3af; margin-bottom:4px;">{tags_str}</div>
                    <div style="font-size:13px;">
                        <span style="color:#ef4444; font-weight:600;">❤ {likes_str}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Use Template button
                if st.button(f"Use This Template", key=f"use_{tmpl['id']}", use_container_width=True, type="primary"):
                    st.session_state["chosen_template_id"] = tmpl["id"]
                    st.session_state["page"] = "builder"
                    st.rerun()

                st.markdown("")  # spacer


# ============================================================
# ---- 11. PAGE: RESUME BUILDER ----
# ============================================================
elif st.session_state["page"] == "builder":
    current_template = next((t for t in TEMPLATES if t["id"] == st.session_state["chosen_template_id"]), TEMPLATES[0])

    # Builder Header
    st.markdown(f"""
    <div class="builder-header">
        <h2>✏️ Building: {current_template['name']}</h2>
        <p>{current_template['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ---- Old Resume Upload ----
    with st.expander("📤 Have an Existing Resume? Upload to auto-fill & upgrade instantly", expanded=False):
        st.markdown("Upload your existing resume in **PDF**, **DOCX**, or **TXT** format. Our AI will automatically extract your info into the chosen template.")
        uploaded_old_resume = st.file_uploader(
            "Upload Existing Resume (.pdf, .docx, .txt)",
            type=["pdf", "docx", "txt"],
            key="old_resume_uploader"
        )

        if uploaded_old_resume is not None:
            if st.button("⚡ Extract & Customize Into Selected Template", type="primary", key="btn_parse_old"):
                with st.spinner("Extracting text and analyzing resume structure with AI..."):
                    raw_extracted, err = extract_text_from_file(uploaded_old_resume)
                    if err:
                        st.error(err)
                    elif raw_extracted:
                        parsed = parse_resume_with_ai(raw_extracted)
                        if parsed:
                            st.session_state["input_name"] = parsed.get("name", "")
                            st.session_state["input_title"] = parsed.get("title", "")
                            st.session_state["input_email"] = parsed.get("email", "")
                            st.session_state["input_phone"] = parsed.get("phone", "")
                            st.session_state["input_address"] = parsed.get("location", "")
                            st.session_state["input_links"] = parsed.get("links", "")
                            st.session_state["input_skills"] = parsed.get("skills", "")
                            st.session_state["input_experience"] = parsed.get("experience", "")
                            st.session_state["input_projects"] = parsed.get("projects", "")
                            st.session_state["input_education"] = parsed.get("education", "")

                            cand_name = parsed.get("name", "Candidate")
                            prompt = (
                                f"You are a top-tier executive resume writer.\n"
                                f"Write an impressive, modern, highly competitive professional resume for this candidate based on their uploaded profile.\n\n"
                                f"Profile:\n"
                                f"- Full Name: {cand_name}\n"
                                f"- Target Role: {parsed.get('title')}\n"
                                f"- Technical Skills: {parsed.get('skills')}\n"
                                f"- Experience: {parsed.get('experience')}\n"
                                f"- Projects: {parsed.get('projects')}\n"
                                f"- Education: {parsed.get('education')}\n\n"
                                f"Guidelines:\n"
                                f"1. Use '## Professional Summary', '## Technical Skills', '## Professional Experience', '## Key Projects', '## Education'.\n"
                                f"2. Use '### Role | Company (Duration)' and bold action verbs for bullets.\n"
                                f"3. Return only the final resume markdown without conversational remarks."
                            )
                            draft = ask_text_ai(prompt)
                            if draft:
                                st.session_state["resume_text"] = draft
                                st.session_state["resume_log"] = [f"Imported from {uploaded_old_resume.name}"]
                                contact_items = [p.strip() for p in (parsed.get("email"), parsed.get("phone"), parsed.get("location"), parsed.get("links")) if p and p.strip()]
                                st.session_state["resume_header"] = {
                                    "name": cand_name,
                                    "title": parsed.get("title", ""),
                                    "contact": "   |   ".join(contact_items),
                                }
                                st.success(f"🎉 Successfully imported into **{current_template['name']}** template!")
                                st.rerun()
                            else:
                                st.error(BUSY_MSG)
                        else:
                            st.error("AI could not extract structured fields. Please fill in the form below manually.")

    # ---- Photo Upload ----
    st.markdown("---")
    st.markdown("#### 📸 Profile Photo (Optional)")
    col_p1, col_p2 = st.columns([1, 3])
    with col_p1:
        if st.session_state.get("photo_bytes"):
            st.image(st.session_state["photo_bytes"], caption="Profile Photo", width=120)
            if st.button("🗑️ Remove Photo", key="btn_remove_photo"):
                st.session_state["photo_bytes"] = None
                st.rerun()
        else:
            st.markdown("""
            <div style="border: 2px dashed #93c5fd; border-radius: 12px; padding: 24px; text-align: center; color: #3b82f6;">
                📸 <b>No Photo</b><br><small>Upload below</small>
            </div>
            """, unsafe_allow_html=True)

    with col_p2:
        photo_file = st.file_uploader(
            "Upload a professional headshot (JPG / PNG)",
            type=["jpg", "jpeg", "png"],
            key="photo_uploader_widget"
        )
        if photo_file is not None:
            st.session_state["photo_bytes"] = photo_file.getvalue()
            st.session_state["include_photo"] = True

        if st.session_state.get("photo_bytes"):
            st.session_state["include_photo"] = st.checkbox(
                "Include photo in the generated PDF",
                value=st.session_state.get("include_photo", True)
            )

    # ---- Candidate Info Form ----
    st.markdown("---")
    st.markdown(f"#### ✏️ Candidate Details for **{current_template['name']}**")

    with st.form("resume_input_form"):
        st.markdown("**👤 Personal & Contact Info**")
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", value=st.session_state.get("input_name", ""), placeholder="e.g. Farooq Khan")
            job_title = st.text_input("Target Job Title", value=st.session_state.get("input_title", ""), placeholder="e.g. Senior AI Developer")
        with col2:
            email = st.text_input("Email", value=st.session_state.get("input_email", ""), placeholder="e.g. farooq@gmail.com")
            phone = st.text_input("Phone", value=st.session_state.get("input_phone", ""), placeholder="e.g. +91 98765 43210")

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            address = st.text_input("City / Location", value=st.session_state.get("input_address", ""), placeholder="e.g. Bhopal, India")
        with col_l2:
            links = st.text_input("LinkedIn / GitHub", value=st.session_state.get("input_links", ""), placeholder="e.g. linkedin.com/in/yourname")

        st.markdown("**🛠️ Technical Skills**")
        skills = st.text_area("Skills (comma-separated)", value=st.session_state.get("input_skills", ""),
                              placeholder="Python, AI, Machine Learning, SQL, Docker, REST APIs")

        st.markdown("**💼 Work Experience**")
        experience = st.text_area("Work experience & roles", value=st.session_state.get("input_experience", ""),
                                  placeholder="2 years as Python Developer building AI workflows and APIs")

        st.markdown("**🚀 Key Projects**")
        projects = st.text_area("Projects & Achievements", value=st.session_state.get("input_projects", ""),
                                placeholder="Built AI Resume Platform with ATS scoring and PDF generation")

        st.markdown("**🎓 Education**")
        education = st.text_area("Education & Degrees", value=st.session_state.get("input_education", ""),
                                 placeholder="BCA - Barkatullah University, 2023")

        with st.expander("📌 Traditional Bio-data Details (Optional)"):
            col3, col4 = st.columns(2)
            with col3:
                father_name = st.text_input("Father's name", value=st.session_state.get("input_father", ""))
                dob = st.text_input("Date of birth", value=st.session_state.get("input_dob", ""), placeholder="05 July 2000")
            with col4:
                gender = st.selectbox("Gender", ["Not specified", "Male", "Female"], index=0)
                marital = st.selectbox("Marital status", ["Not specified", "Single", "Married"], index=0)

        btn_submit = st.form_submit_button(f"✨ Build My Resume", type="primary")

    # Save fields
    st.session_state["input_name"] = name
    st.session_state["input_title"] = job_title
    st.session_state["input_email"] = email
    st.session_state["input_phone"] = phone
    st.session_state["input_address"] = address
    st.session_state["input_links"] = links
    st.session_state["input_skills"] = skills
    st.session_state["input_experience"] = experience
    st.session_state["input_projects"] = projects
    st.session_state["input_education"] = education

    # ---- Form Submission ----
    if btn_submit:
        if not name.strip():
            st.warning("Please type your name first!")
        else:
            with st.spinner(f"AI is crafting your resume in {current_template['name']} template..."):
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
                    f"   - '## Professional Summary': A 3-4 sentence powerful summary.\n"
                    f"   - '## Technical Skills': Categorize into 3-4 grouped bullets.\n"
                    f"   - '## Professional Experience': Use '### Role | Company (Duration)' with 3-4 achievement bullets.\n"
                    f"   - '## Key Projects': 2 standout technical projects.\n"
                    f"   - '## Education': Degree, university, year.\n"
                    f"   {f'- ## Personal Details: Include if specified.' if (father_name.strip() or dob.strip()) else 'DO NOT include personal bio-data.'}\n"
                    f"2. Ensure content fills a standard 1-page executive resume.\n"
                    f"3. NEVER use placeholder brackets. Output ONLY the final resume markdown."
                )
                draft = ask_text_ai(instructions)
                if not draft:
                    st.error(BUSY_MSG)
                else:
                    st.session_state["resume_text"] = draft
                    st.session_state["resume_log"] = [f"Generated in {current_template['name']}"]
                    st.session_state["resume_header"] = {
                        "name": name.strip(),
                        "title": job_title.strip(),
                        "email": email.strip(),
                        "phone": phone.strip(),
                        "contact": "   |   ".join(contact_parts),
                    }

    # ---- Revision Chat ----
    change = st.chat_input("Want a change? Type it here (e.g. add more cloud skills, expand projects)")
    if change:
        if not st.session_state["resume_text"]:
            st.warning("Please build your resume first, then ask for revisions.")
        else:
            with st.spinner("AI is applying your revisions..."):
                ask = (
                    f"Here is my current resume:\n\n{st.session_state['resume_text']}\n\n"
                    f"My revision request: {change}\n\n"
                    f"Rules:\n"
                    f"1. Apply the exact change requested. Keep all other facts intact.\n"
                    f"2. Keep the same format: '## Section' headings, '### Subheading' for roles, '-' bullets.\n"
                    f"3. Fix spelling and grammar.\n"
                    f"4. Never write notes or [brackets].\n"
                    f"5. Return the full updated resume text only."
                )
                updated = ask_text_ai(ask)
                if not updated:
                    st.error(BUSY_MSG)
                else:
                    st.session_state["resume_text"] = updated
                    st.session_state["resume_log"].append(change)

    # ---- Resume Preview & Export ----
    if st.session_state["resume_text"]:
        st.markdown("---")
        tab_view, tab_ats = st.tabs(["📄 Resume Preview & Export", "🎯 ATS Scanner & Optimizer"])

        with tab_view:
            header = st.session_state.get("resume_header") or {}
            safe_name = (header.get("name") or "My_Resume").replace(" ", "_")

            # Template & theme switcher
            col_c1, col_c2, col_c3 = st.columns([1, 1, 1])
            with col_c1:
                template_names = [t["name"] for t in TEMPLATES]
                cur_idx = template_names.index(current_template["name"]) if current_template["name"] in template_names else 0
                selected_tmpl_name = st.selectbox("Switch Template:", template_names, index=cur_idx, key="live_tmpl_select")
                matched_tmpl = next((t for t in TEMPLATES if t["name"] == selected_tmpl_name), current_template)
                if matched_tmpl["id"] != st.session_state["chosen_template_id"]:
                    st.session_state["chosen_template_id"] = matched_tmpl["id"]
                    st.rerun()

            with col_c2:
                theme_names = list(RESUME_THEMES.keys())
                def_theme_idx = theme_names.index(current_template["theme"]) if current_template["theme"] in theme_names else 0
                chosen_theme = st.selectbox("Color Palette:", theme_names, index=def_theme_idx, key="live_theme_select")

            with col_c3:
                layout_options = [
                    "Modern Clean (Single Column)",
                    "Modern Sidebar (Dual Tone)",
                    "Executive Banner (Top Accent)"
                ]
                def_layout_idx = layout_options.index(current_template["layout"]) if current_template["layout"] in layout_options else 0
                chosen_layout = st.selectbox("Layout:", layout_options, index=def_layout_idx, key="live_layout_select")

            # Generate PDF
            photo_to_use = st.session_state.get("photo_bytes") if st.session_state.get("include_photo") else None
            pdf_bytes = make_resume_pdf(
                st.session_state["resume_text"], header,
                theme=chosen_theme, layout=chosen_layout, photo_bytes=photo_to_use
            )

            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button(
                    "📥 Download PDF", pdf_bytes, key="resume_pdf_dl",
                    file_name=f"Resume_{safe_name}.pdf",
                    mime="application/pdf", use_container_width=True, type="primary"
                )
            with col_b:
                st.download_button(
                    "📥 Download TXT",
                    st.session_state["resume_text"].encode("utf-8"),
                    key="resume_txt_dl",
                    file_name=f"Resume_{safe_name}.txt",
                    mime="text/plain", use_container_width=True
                )

            st.markdown("---")
            st.markdown(f"### Preview: {current_template['name']}")
            if photo_to_use:
                p_col1, p_col2 = st.columns([1, 5])
                with p_col1:
                    st.image(photo_to_use, width=100, caption="Profile Photo")
                with p_col2:
                    st.markdown(st.session_state["resume_text"])
            else:
                st.markdown(st.session_state["resume_text"])

            if st.session_state.get("resume_log"):
                st.caption("Change Log:  " + "  •  ".join(st.session_state["resume_log"]))

        with tab_ats:
            st.markdown("#### 🎯 ATS Readiness Check")
            st.write("Over 75% of companies filter candidates using automated ATS scanners.")

            target_jd = st.text_area(
                "Target Job Description (Optional)",
                placeholder="Paste job requirements here...",
                height=90, key="ats_jd_input"
            )

            if st.button("⚡ Scan ATS Score", key="ats_scan_btn", type="primary"):
                with st.spinner("Analyzing keyword density and ATS readability..."):
                    ats_instructions = (
                        f"You are a Fortune 500 Senior Technical Recruiter and ATS algorithm auditor.\n"
                        f"Analyze this resume:\n\n{st.session_state['resume_text']}\n\n"
                        f"{f'Target Role / Requirements: {target_jd}' if target_jd.strip() else 'Target: Standard industry ATS benchmark'}\n\n"
                        f"Output your report strictly in this format:\n"
                        f"SCORE: [Integer between 65 and 96]\n"
                        f"### 🌟 ATS Strengths\n"
                        f"- [2-3 bullet points]\n\n"
                        f"### ⚠️ Missing Keywords & Gaps\n"
                        f"- [2-3 bullet points]\n\n"
                        f"### 💡 High-Impact Recommendations\n"
                        f"- [2-3 specific suggestions]"
                    )
                    ats_report = ask_text_ai(ats_instructions)
                    if ats_report:
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
                    st.metric("ATS Score", f"{sc} / 100")
                with c_sc2:
                    st.progress(sc / 100)
                    if sc >= 85:
                        st.success("🟢 Strong ATS match!")
                    elif sc >= 70:
                        st.info("🟡 Moderate match — needs more keywords.")
                    else:
                        st.warning("🔴 Low match — risk of rejection.")

                st.markdown(st.session_state["ats_report"])

                st.markdown("---")
                if st.button("✨ Auto-Optimize for 95+ ATS Score", key="ats_auto_opt_btn"):
                    with st.spinner("Upgrading with power verbs and metrics..."):
                        opt_prompt = (
                            f"You are an executive resume optimizer. Upgrade this resume to achieve a 95+ ATS score.\n\n"
                            f"Original Resume:\n{st.session_state['resume_text']}\n\n"
                            f"{f'Target Job Requirements: {target_jd}' if target_jd.strip() else ''}\n\n"
                            f"Rules:\n"
                            f"1. Replace passive verbs with strong action verbs.\n"
                            f"2. Add realistic metrics without fabricating.\n"
                            f"3. Maintain '## Section' headings and '-' bullet formatting.\n"
                            f"4. Keep all factual details accurate.\n"
                            f"5. Output ONLY the upgraded resume text."
                        )
                        upgraded_res = ask_text_ai(opt_prompt)
                        if upgraded_res:
                            st.session_state["resume_text"] = upgraded_res
                            st.session_state["resume_log"].append("Auto-optimized for 95+ ATS")
                            st.session_state["ats_score"] = 96
                            st.success("🎉 Resume auto-optimized! Check the Preview tab.")
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)
    else:
        st.info("👆 Fill your details above and press **Build My Resume** to generate your professional PDF.")
