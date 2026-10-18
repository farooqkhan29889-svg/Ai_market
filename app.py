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

# ---- 0b. Supabase Auth + Saved Resumes (optional backend) ----
# Works only when SUPABASE_URL and SUPABASE_ANON_KEY are set (in .env locally
# or Streamlit Secrets on Cloud). If they are missing, the app still runs and
# the auth/save UI shows a friendly "not configured yet" notice instead.
try:
    from supabase import create_client
except ImportError:
    create_client = None

def _sb_keys():
    return os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_ANON_KEY")

def sb_enabled():
    url, key = _sb_keys()
    return bool(url and key and create_client is not None)

def _sb():
    """Return a per-session Supabase client, restoring the stored login session."""
    if not sb_enabled():
        return None
    client = st.session_state.get("_sb_client")
    if client is None:
        url, key = _sb_keys()
        client = create_client(url, key)
        st.session_state["_sb_client"] = client
    sess = st.session_state.get("sb_session")
    if sess:
        try:
            client.auth.set_session(sess["access_token"], sess["refresh_token"])
        except Exception:
            st.session_state.pop("sb_session", None)
    return client

def _store_session(session):
    if session and getattr(session, "access_token", None):
        st.session_state["sb_session"] = {
            "access_token": session.access_token,
            "refresh_token": session.refresh_token,
        }

def sb_current_user():
    """Return the logged-in user object (with .email/.id) or None."""
    if not sb_enabled():
        return None
    try:
        return _sb().auth.get_user().user
    except Exception:
        return None

def sb_sign_in(email, password):
    res = _sb().auth.sign_in_with_password({"email": email, "password": password})
    _store_session(res.session)
    return res.user

def sb_sign_up(email, password, full_name=None):
    payload = {"email": email, "password": password}
    if full_name:
        payload["options"] = {"data": {"full_name": full_name}}
    res = _sb().auth.sign_up(payload)
    if res.session:
        _store_session(res.session)
    return res.user, res.session

def sb_sign_out():
    try:
        _sb().auth.sign_out()
    except Exception:
        pass
    st.session_state.pop("sb_session", None)

def sb_save_resume(user_id, name, resume_text, header, template_id):
    _sb().table("resumes").insert({
        "user_id": user_id,
        "name": name,
        "resume_text": resume_text,
        "header_json": json.dumps(header or {}),
        "template_id": template_id,
    }).execute()

def sb_list_resumes(user_id):
    rows = (_sb().table("resumes").select("*")
            .eq("user_id", user_id).order("created_at", desc=True).execute())
    return rows.data or []

def sb_delete_resume(user_id, resume_id):
    _sb().table("resumes").delete().eq("id", resume_id).eq("user_id", user_id).execute()

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
        "accent_color": "#334155",
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
    a, a:link, a:visited { color: #0d9488; text-decoration: none; }
    a:hover { color: #0f766e; text-decoration: underline; }

    /* ── Page background: soft canvas so white cards & preview pop ── */
    [data-testid="stAppViewContainer"] { background: #f8fafc; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stMain"] { background: transparent; }
    [data-testid="stAppViewContainer"] > .main,
    [data-testid="stMainBlockContainer"] { background: transparent; }

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
        font-size: 24px; font-weight: 800; color: #1e293b;
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
        position: relative; overflow: hidden;
        background:
          radial-gradient(1100px 380px at 88% -12%, rgba(180, 83, 9, 0.55), transparent 60%),
          radial-gradient(900px 480px at 8% 128%, rgba(13, 148, 136, 0.55), transparent 55%),
          linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px; padding: 40px 44px 30px;
        margin-bottom: 30px; color: #ffffff;
        box-shadow: 0 18px 50px rgba(15, 23, 42, 0.28);
    }
    .hero-banner h1 {
        font-size: 38px; font-weight: 900; margin: 0 0 8px;
        letter-spacing: -0.8px; line-height: 1.08;
    }
    .hero-banner p {
        font-size: 15px; opacity: 0.9; margin: 0; line-height: 1.55;
        max-width: 620px;
    }
    .hero-stats-bar {
        display: flex; gap: 14px; margin-top: 24px;
        flex-wrap: wrap;
    }
    .hero-stat {
        text-align: center; min-width: 96px;
        background: rgba(255, 255, 255, 0.07);
        border: 1px solid rgba(255, 255, 255, 0.14);
        border-radius: 14px; padding: 12px 18px;
        backdrop-filter: blur(6px);
    }
    .hero-stat .num {
        font-size: 24px; font-weight: 800; display: block;
        background: linear-gradient(135deg, #fbbf24, #f59e0b);
        -webkit-background-clip: text; background-clip: text;
        -webkit-text-fill-color: transparent; color: #f59e0b;
    }
    .hero-stat .lbl {
        font-size: 10px; text-transform: uppercase;
        letter-spacing: 1px; opacity: 0.85;
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
        background: #1e293b; color: #ffffff;
        border-color: #1e293b;
    }

    /* ── Template Cards ── */
    .tmpl-card {
        position: relative;
        background: #ffffff; border: 1px solid #e8ebf0;
        border-radius: 14px; overflow: hidden;
        transition: all 0.25s ease;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.05);
    }
    .tmpl-card::before {
        content: ""; position: absolute; top: 0; left: 0; right: 0;
        height: 3px; z-index: 2;
        background: linear-gradient(90deg, #0d9488, #b45309);
        transform: scaleX(0); transform-origin: left;
        transition: transform 0.3s ease;
    }
    .tmpl-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 16px 34px rgba(15, 23, 42, 0.14);
        border-color: #cbd5e1;
    }
    .tmpl-card:hover::before { transform: scaleX(1); }
    .tmpl-img-wrap {
        position: relative;
        background: linear-gradient(180deg, #f8fafc, #f1f5f9);
        padding: 14px; display: flex; justify-content: center;
        min-height: 220px; align-items: center;
    }
    .tmpl-img-wrap img {
        max-height: 200px; border-radius: 6px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.12);
    }
    .tmpl-overlay {
        position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: rgba(15, 23, 42, 0.7);
        display: flex; align-items: center; justify-content: center;
        opacity: 0; transition: opacity 0.25s; border-radius: 0;
    }
    .tmpl-card:hover .tmpl-overlay { opacity: 1; }
    .tmpl-overlay-btn {
        background: #ffffff; color: #1e293b;
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
        padding-left: 12px; border-left: 4px solid #0d9488;
    }
    .section-sub {
        font-size: 13.5px; color: #6b7280;
        margin-bottom: 18px; line-height: 1.4;
    }

    /* ── Buttons ── */
    div[data-testid="stButton"] button[kind="primary"],
    div[data-testid="stDownloadButton"] button[kind="primary"] {
        background: linear-gradient(135deg, #1e293b, #b45309);
        color: #ffffff; border: none; border-radius: 8px;
        font-weight: 700; padding: 0.5rem 1.2rem;
        transition: transform 0.15s, box-shadow 0.15s;
    }
    div[data-testid="stButton"] button[kind="primary"]:hover,
    div[data-testid="stDownloadButton"] button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(180, 83, 9, 0.35);
        color: #ffffff;
    }

    /* ── Builder page ── */
    .builder-header {
        background: linear-gradient(135deg, #0f172a, #1e293b);
        border-radius: 12px; padding: 24px 30px; color: #fff;
        margin-bottom: 20px;
    }
    .builder-header h2 { margin: 0 0 4px; font-weight: 800; }
    .builder-header p { margin: 0; opacity: 0.85; font-size: 14px; }

    /* Chat Input */
    div[data-testid="stChatInput"] {
        border: 2px solid #99f6e4; border-radius: 12px;
    }

    /* Hide Streamlit footer */
    footer { visibility: hidden; }

    /* ── AI Tools Page ── */
    .ai-tools-banner {
        background: linear-gradient(135deg, #f0fdfa 0%, #f1f5f9 55%, #fffbeb 100%);
        border: 1px solid #e2e8f0;
        border-radius: 14px; padding: 18px 28px;
        margin-bottom: 28px;
        display: flex; align-items: center; justify-content: space-between;
        flex-wrap: wrap; gap: 12px;
    }
    .ai-tools-banner .banner-text {
        font-size: 14.5px; color: #0f172a;
        font-weight: 500;
    }
    .ai-tools-banner .upgrade-btn {
        background: linear-gradient(135deg, #b45309, #d97706);
        color: #fff; border: none; border-radius: 24px;
        padding: 10px 28px; font-weight: 700; font-size: 14px;
        cursor: pointer; text-decoration: none;
        box-shadow: 0 4px 14px rgba(180, 83, 9, 0.3);
        transition: transform 0.15s, box-shadow 0.15s;
    }
    .ai-tools-banner .upgrade-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(180, 83, 9, 0.4);
    }
    .ai-tools-heading {
        font-size: 28px; font-weight: 800; color: #111827;
        margin: 0 0 24px; letter-spacing: -0.4px;
    }
    .ai-tool-card {
        background: #ffffff; border: 1.5px solid #e5e7eb;
        border-radius: 14px; padding: 22px 24px;
        margin-bottom: 14px;
        display: flex; align-items: center;
        justify-content: space-between;
        transition: all 0.25s ease;
        cursor: default;
        gap: 16px;
    }
    .ai-tool-card:hover {
        border-color: #0d9488;
        box-shadow: 0 6px 24px rgba(13, 148, 136, 0.14);
        transform: translateY(-2px);
    }
    .ai-tool-left {
        display: flex; align-items: center; gap: 16px;
        flex: 1;
    }
    .ai-tool-icon {
        width: 44px; height: 44px;
        border-radius: 12px;
        display: flex; align-items: center; justify-content: center;
        font-size: 22px; flex-shrink: 0;
    }
    .ai-tool-icon.purple { background: #f1f5f9; }
    .ai-tool-icon.blue { background: #f1f5f9; }
    .ai-tool-icon.green { background: #d1fae5; }
    .ai-tool-icon.orange { background: #ffedd5; }
    .ai-tool-icon.pink { background: #fce7f3; }
    .ai-tool-icon.teal { background: #ccfbf1; }
    .ai-tool-icon.red { background: #fee2e2; }
    .ai-tool-icon.indigo { background: #f1f5f9; }
    .ai-tool-icon.yellow { background: #fef9c3; }
    .ai-tool-info h4 {
        font-size: 15.5px; font-weight: 700; color: #111827;
        margin: 0 0 3px; display: flex; align-items: center; gap: 8px;
    }
    .ai-tool-info p {
        font-size: 13px; color: #6b7280;
        margin: 0; line-height: 1.45;
    }
    .new-badge {
        background: linear-gradient(135deg, #b45309, #1e293b);
        color: #fff; font-size: 10px; font-weight: 700;
        padding: 2px 8px; border-radius: 10px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .hot-badge {
        background: linear-gradient(135deg, #0d9488, #0f766e);
        color: #fff; font-size: 10px; font-weight: 700;
        padding: 2px 8px; border-radius: 10px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }
    .pro-badge {
        background: linear-gradient(135deg, #b45309, #d97706);
        color: #fff; font-size: 10px; font-weight: 700;
        padding: 2px 8px; border-radius: 10px;
        text-transform: uppercase; letter-spacing: 0.5px;
    }

    /* AI Tool Result Box */
    .ai-result-box {
        background: #f8fafc; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 20px 24px;
        margin: 12px 0 20px; line-height: 1.7;
        font-size: 14px; color: #1e293b;
    }
    .ai-result-box h3, .ai-result-box h4 {
        color: #1e293b; margin-top: 12px;
    }

    /* Sidebar AI Tools nav button */
    .sidebar-ai-btn {
        background: linear-gradient(135deg, #b45309, #1e293b);
        color: #ffffff !important; border: none; border-radius: 10px;
        padding: 10px 16px; font-weight: 700; font-size: 14px;
        width: 100%; text-align: center; cursor: pointer;
        margin: 8px 0; display: block;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.25);
        transition: transform 0.15s, box-shadow 0.15s;
        text-decoration: none;
    }
    .sidebar-ai-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(15, 23, 42, 0.35);
    }

    /* Tool active panel */
    .tool-panel-header {
        background: linear-gradient(135deg, #334155 0%, #1e293b 100%);
        border-radius: 12px; padding: 20px 24px;
        color: #ffffff; margin-bottom: 20px;
    }
    .tool-panel-header h3 {
        margin: 0 0 4px; font-weight: 800; font-size: 20px; color: #ffffff;
    }
    .tool-panel-header p {
        margin: 0; opacity: 0.9; font-size: 13.5px; color: #e2e8f0;
    }

    /* ── Top Navigation Bar ── */
    div[class*="st-key-nav_brand"] button,
    div[class*="st-key-navlink_"] button {
        background: transparent !important; border: none !important;
        color: #334155 !important; font-weight: 600; font-size: 15px;
        box-shadow: none !important; padding: 0.3rem 0.5rem !important;
        min-height: 0; letter-spacing: -0.2px;
    }
    div[class*="st-key-nav_brand"] button {
        color: #0f172a !important; font-weight: 800; font-size: 19px;
    }
    div[class*="st-key-nav_brand"] button:hover,
    div[class*="st-key-navlink_"] button:hover {
        color: #b45309 !important; background: transparent !important;
        box-shadow: none !important; text-decoration: none !important;
    }
    div[class*="st-key-nav_signin"] button {
        background: transparent !important; border: 1.5px solid #1e293b !important;
        color: #1e293b !important; border-radius: 22px !important; font-weight: 700;
        box-shadow: none !important; padding: 0.3rem 1.1rem !important; min-height: 0;
    }
    div[class*="st-key-nav_signin"] button:hover {
        background: #1e293b !important; color: #ffffff !important;
    }
    div[class*="st-key-nav_signup"] button {
        background: linear-gradient(135deg, #b45309, #d97706) !important;
        border: none !important; color: #ffffff !important; border-radius: 22px !important;
        font-weight: 700; box-shadow: 0 4px 14px rgba(180, 83, 9, 0.28) !important;
        padding: 0.3rem 1.1rem !important; min-height: 0;
    }
    div[class*="st-key-nav_signup"] button:hover {
        transform: translateY(-2px); box-shadow: 0 6px 18px rgba(180, 83, 9, 0.4) !important;
    }
    .topnav-rule { border: none; border-top: 1px solid #e2e8f0; margin: 2px 0 18px; }
    .nav-user {
        color: #334155; font-weight: 600; font-size: 14px;
        padding: 0.35rem 0.2rem; white-space: nowrap; overflow: hidden;
        text-overflow: ellipsis;
    }
    div[class*="st-key-nav_signout"] button {
        background: transparent !important; border: 1.5px solid #cbd5e1 !important;
        color: #475569 !important; border-radius: 22px !important; font-weight: 700;
        box-shadow: none !important; padding: 0.3rem 0.9rem !important; min-height: 0;
    }
    div[class*="st-key-nav_signout"] button:hover {
        border-color: #b45309 !important; color: #b45309 !important; background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# ---- 8. Session State Initialization ----
defaults = {
    "page": "gallery",  # "gallery", "builder", or "ai_tools"
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
    # AI Tools state
    "ai_tool_active": None,
    "ai_tool_result": None,
    "translate_lang": "Hindi",
    "cover_letter_jd": "",
    "interview_role": "",
    "interview_questions": None,
    "linkedin_result": None,
    "skill_gap_jd": "",
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

# ---- Live resume preview (right pane on the AI Tools page) ----
def render_live_resume_preview():
    st.markdown("### 📄 Live Preview")
    rt = (st.session_state.get("resume_text") or "").strip()
    if not rt:
        st.info("Build a resume first — your live PDF preview will appear here while you use the AI tools.")
        return
    header = st.session_state.get("resume_header") or {}
    tmpl_id = st.session_state.get("chosen_template_id")
    tmpl = next((t for t in TEMPLATES if t["id"] == tmpl_id), TEMPLATES[0])
    theme = st.session_state.get("live_theme_select") or tmpl.get("theme", "Modern Navy")
    layout = st.session_state.get("live_layout_select") or tmpl.get("layout", "Modern Clean (Single Column)")
    photo = st.session_state.get("photo_bytes") if st.session_state.get("include_photo", True) else None
    if pdfium is None:
        st.info("Install pypdfium2 to see the live visual preview.")
        return
    try:
        pdf_bytes = make_resume_pdf(rt, header, theme=theme, layout=layout, photo_bytes=photo)
        doc = pdfium.PdfDocument(pdf_bytes)
        for i in range(len(doc)):
            st.image(doc[i].render(scale=1.5).to_pil(), use_container_width=True)
    except Exception as e:
        st.error(f"Could not render preview: {e}")

# ============================================================
# ---- 8b. TOP NAVIGATION BAR + AUTH / INFO DIALOGS ----
# ============================================================
@st.dialog("Sign in to ResumeStudio")
def _signin_dialog():
    if not sb_enabled():
        st.info("Accounts aren't set up yet. The developer needs to add SUPABASE_URL and "
                "SUPABASE_ANON_KEY to the app's secrets to enable sign in.")
        return
    with st.form("signin_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        ok = st.form_submit_button("Sign in", type="primary", use_container_width=True)
    if ok:
        try:
            sb_sign_in(email.strip(), password)
            st.success("Signed in — welcome back!")
            time.sleep(0.4)
            st.rerun()
        except Exception as e:
            st.error(f"Sign in failed: {e}")

@st.dialog("Create your free account")
def _signup_dialog():
    if not sb_enabled():
        st.info("Accounts aren't set up yet. The developer needs to add SUPABASE_URL and "
                "SUPABASE_ANON_KEY to the app's secrets to enable sign up.")
        return
    with st.form("signup_form"):
        full_name = st.text_input("Full name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        ok = st.form_submit_button("Create account", type="primary", use_container_width=True)
    if ok:
        try:
            user, session = sb_sign_up(email.strip(), password, full_name.strip())
            if session:
                st.success("Account created! You're signed in.")
                time.sleep(0.5)
                st.rerun()
            else:
                st.success("Account created! Check your email to confirm, then sign in.")
        except Exception as e:
            st.error(f"Sign up failed: {e}")

@st.dialog("Pricing")
def _pricing_dialog():
    st.markdown("**Everything is 100% free right now.**")
    st.markdown(
        "- Unlimited resumes, templates & PDF export\n"
        "- All 10 AI tools included\n"
        "- No sign-up required"
    )
    st.caption("A paid Pro tier is planned but not built yet.")

@st.dialog("Help & Support")
def _help_dialog():
    st.markdown("**Getting started:** pick a template, fill your details, press **Build My Resume**, then download the PDF.")
    st.markdown("Use **AI Tools** to translate, improve, tailor your resume, or draft a cover letter.")
    st.caption("Add a contact / support link here when you have one.")

def render_top_nav():
    user = sb_current_user()
    brand, links, auth = st.columns([2.4, 5.6, 2.4], gap="small")
    with brand:
        if st.button("📄 ResumeStudio", key="nav_brand"):
            st.session_state["page"] = "gallery"
            st.rerun()
    with links:
        if user:
            c1, c2, c3, c4, c5, _ = st.columns([1, 1, 1.2, 1, 1, 1])
        else:
            c1, c2, c3, c4, _ = st.columns([1, 1, 1, 1, 2])
        if c1.button("Templates", key="navlink_templates"):
            st.session_state["page"] = "gallery"
            st.rerun()
        if c2.button("AI Tools", key="navlink_aitools"):
            st.session_state["page"] = "ai_tools"
            st.session_state["ai_tool_active"] = None
            st.session_state["ai_tool_result"] = None
            st.rerun()
        if user:
            if c3.button("My Resumes", key="navlink_mine"):
                st.session_state["page"] = "my_resumes"
                st.rerun()
            if c4.button("Pricing", key="navlink_pricing"):
                _pricing_dialog()
            if c5.button("Help", key="navlink_help"):
                _help_dialog()
        else:
            if c3.button("Pricing", key="navlink_pricing"):
                _pricing_dialog()
            if c4.button("Help", key="navlink_help"):
                _help_dialog()
    with auth:
        if user:
            a1, a2 = st.columns([1.6, 1])
            a1.markdown(
                f'<div class="nav-user">👤 {(getattr(user, "email", "") or "Signed in")[:22]}</div>',
                unsafe_allow_html=True,
            )
            if a2.button("Sign out", key="nav_signout"):
                sb_sign_out()
                st.session_state["page"] = "gallery"
                st.rerun()
        else:
            a1, a2 = st.columns(2)
            if a1.button("Sign in", key="nav_signin"):
                _signin_dialog()
            if a2.button("Sign up", key="nav_signup"):
                _signup_dialog()
    st.markdown('<hr class="topnav-rule">', unsafe_allow_html=True)

# ============================================================
# ---- 9. SIDEBAR: Categories & Branding ----
# ============================================================
with st.sidebar:
    st.markdown('<div class="sidebar-brand">📄 ResumeStudio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Professional Resume Templates<br>Free Builder • ATS-Optimized • PDF Export</div>', unsafe_allow_html=True)

    # AI Tools navigation
    if st.button("✨ AI Tools", key="nav_ai_tools", use_container_width=True, type="primary"):
        st.session_state["page"] = "ai_tools"
        st.session_state["ai_tool_active"] = None
        st.session_state["ai_tool_result"] = None
        st.rerun()

    # Back to gallery button (when in builder or ai_tools)
    if st.session_state["page"] in ("builder", "ai_tools"):
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


# ---- Top navigation bar (shown on every page) ----
render_top_nav()

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
            <div style="border: 2px dashed #cbd5e1; border-radius: 12px; padding: 24px; text-align: center; color: #b45309;">
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

            # Save to account (requires sign-in)
            _sb_user = sb_current_user()
            if not sb_enabled():
                st.caption("💾 Cloud saving unlocks once the developer connects Supabase (SUPABASE_URL / SUPABASE_ANON_KEY).")
            elif _sb_user is None:
                st.info("Sign in to save this resume to your account and access it from **My Resumes**.")
            else:
                sc1, sc2 = st.columns([3, 1])
                save_name = sc1.text_input("Save as", value=f"{safe_name} Resume", key="save_resume_name")
                if sc2.button("💾 Save", type="primary", use_container_width=True, key="save_resume_btn"):
                    try:
                        sb_save_resume(_sb_user.id, save_name.strip() or "Untitled",
                                       st.session_state["resume_text"], header,
                                       st.session_state.get("chosen_template_id"))
                        st.success("Saved to your account!")
                    except Exception as e:
                        st.error(f"Could not save: {e}")

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

# ============================================================
# ---- 12. PAGE: AI TOOLS ----
# ============================================================
elif st.session_state["page"] == "ai_tools":

    # AI Tools definitions
    AI_TOOLS = [
        {
            "id": "translate",
            "icon": "🌐",
            "icon_class": "purple",
            "name": "Translate Resume",
            "badge": "new",
            "desc": "Create a translated version of your resume in the language of your choice.",
            "btn_text": "Translate now",
        },
        {
            "id": "improve_writing",
            "icon": "✏️",
            "icon_class": "pink",
            "name": "Improve Writing",
            "badge": "new",
            "desc": "Strengthen your resume with clearer, more concise and impactful writing.",
            "btn_text": "Improve now",
        },
        {
            "id": "cover_letter",
            "icon": "📝",
            "icon_class": "blue",
            "name": "Draft Cover Letter",
            "badge": "new",
            "desc": "Draft a tailored cover letter based on your resume and target job description.",
            "btn_text": "Draft now",
        },
        {
            "id": "summary",
            "icon": "📋",
            "icon_class": "green",
            "name": "Generate Summary",
            "badge": "new",
            "desc": "Turn your existing resume content into a professional summary for your resume.",
            "btn_text": "Generate now",
        },
        {
            "id": "interview_prep",
            "icon": "🎤",
            "icon_class": "orange",
            "name": "Interview Prep",
            "badge": "hot",
            "desc": "Generate tailored interview questions and best-practice answers for your target role.",
            "btn_text": "Prepare now",
        },
        {
            "id": "linkedin",
            "icon": "💼",
            "icon_class": "blue",
            "name": "LinkedIn Optimizer",
            "badge": "new",
            "desc": "Optimize your LinkedIn headline, summary, and about section from your resume.",
            "btn_text": "Optimize now",
        },
        {
            "id": "skill_gap",
            "icon": "📊",
            "icon_class": "teal",
            "name": "Skill Gap Analyzer",
            "badge": "hot",
            "desc": "Compare your skills against a job description and discover what you're missing.",
            "btn_text": "Analyze now",
        },
        {
            "id": "email_followup",
            "icon": "📧",
            "icon_class": "indigo",
            "name": "Email Follow-Up",
            "badge": "new",
            "desc": "Draft a professional follow-up email after applying or interviewing.",
            "btn_text": "Draft email",
        },
        {
            "id": "tailor_resume",
            "icon": "🎯",
            "icon_class": "red",
            "name": "Tailor Resume to Job",
            "badge": "hot",
            "desc": "Automatically customize your resume keywords and bullets for a specific job posting.",
            "btn_text": "Tailor now",
        },
        {
            "id": "grammar_check",
            "icon": "✅",
            "icon_class": "yellow",
            "name": "Grammar & Style Check",
            "badge": "new",
            "desc": "Find and fix grammar errors, typos, and awkward phrasing in your resume.",
            "btn_text": "Check now",
        },
    ]

    active_tool = st.session_state.get("ai_tool_active")

    # If no tool is active, show the tool listing page
    _tools_col, _preview_col = st.columns([1.7, 1.15], gap="large")

    with _preview_col:
        render_live_resume_preview()

    with _tools_col:
        if active_tool is None:

            # Pro banner
            st.markdown("""
            <div class="ai-tools-banner">
                <span class="banner-text">✨ AI-powered tools to supercharge your job search — all included free!</span>
            </div>
            """, unsafe_allow_html=True)

            # Heading
            st.markdown('<div class="ai-tools-heading">What would you like to do?</div>', unsafe_allow_html=True)

            # Check if user has a resume
            has_resume = bool(st.session_state.get("resume_text", "").strip())

            if not has_resume:
                st.warning("⚠️ **Build your resume first!** Go to Templates → choose a template → fill in your details → Build. Then come back here to use AI tools on your resume.")
                st.markdown("---")

            # Render each tool card
            for tool in AI_TOOLS:
                badge_html = ""
                if tool.get("badge") == "new":
                    badge_html = '<span class="new-badge">New</span>'
                elif tool.get("badge") == "hot":
                    badge_html = '<span class="hot-badge">🔥 Hot</span>'
                elif tool.get("badge") == "pro":
                    badge_html = '<span class="pro-badge">Pro</span>'

                st.markdown(f"""
                <div class="ai-tool-card">
                    <div class="ai-tool-left">
                        <div class="ai-tool-icon {tool['icon_class']}">{tool['icon']}</div>
                        <div class="ai-tool-info">
                            <h4>{tool['name']} {badge_html}</h4>
                            <p>{tool['desc']}</p>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if st.button(tool["btn_text"], key=f"ai_tool_btn_{tool['id']}", disabled=not has_resume, use_container_width=False):
                    st.session_state["ai_tool_active"] = tool["id"]
                    st.session_state["ai_tool_result"] = None
                    st.rerun()

        # ---- Active Tool Panels ----
        else:
            # Back button
            if st.button("← Back to AI Tools", key="back_to_tools"):
                st.session_state["ai_tool_active"] = None
                st.session_state["ai_tool_result"] = None
                st.rerun()

            resume_text = st.session_state.get("resume_text", "")
            header = st.session_state.get("resume_header", {})

            # ---- TRANSLATE RESUME ----
            if active_tool == "translate":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>🌐 Translate Resume</h3>
                    <p>Create a professional translation of your resume in any language</p>
                </div>
                """, unsafe_allow_html=True)

                languages = [
                    "Hindi", "Spanish", "French", "German", "Chinese (Simplified)",
                    "Japanese", "Korean", "Arabic", "Portuguese", "Russian",
                    "Italian", "Dutch", "Turkish", "Bengali", "Tamil",
                    "Telugu", "Marathi", "Gujarati", "Urdu", "Punjabi",
                ]
                selected_lang = st.selectbox("Select target language:", languages, key="translate_lang_select")

                if st.button("🌐 Translate Now", type="primary", key="btn_do_translate"):
                    with st.spinner(f"Translating your resume to {selected_lang}..."):
                        prompt = (
                            f"You are a professional resume translator. Translate the following resume "
                            f"into {selected_lang} while maintaining professional tone, formatting (## headings, ### subheadings, - bullets), "
                            f"and all factual details. Do NOT translate the person's name. "
                            f"Output ONLY the translated resume.\n\n"
                            f"Resume:\n{resume_text}"
                        )
                        result = ask_text_ai(prompt)
                        if result:
                            st.session_state["ai_tool_result"] = result
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Translated Resume")
                    st.markdown(st.session_state["ai_tool_result"])
                    st.download_button(
                        "📥 Download Translated Resume",
                        st.session_state["ai_tool_result"].encode("utf-8"),
                        file_name=f"Resume_Translated_{selected_lang}.txt",
                        mime="text/plain", type="primary"
                    )

            # ---- IMPROVE WRITING ----
            elif active_tool == "improve_writing":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>✏️ Improve Writing</h3>
                    <p>Strengthen your resume with clearer, more powerful language</p>
                </div>
                """, unsafe_allow_html=True)

                improvement_focus = st.multiselect(
                    "Focus areas:",
                    ["Stronger action verbs", "Quantify achievements", "Remove filler words",
                     "More concise bullets", "Professional tone", "Industry keywords"],
                    default=["Stronger action verbs", "Quantify achievements", "More concise bullets"],
                    key="improve_focus"
                )

                if st.button("✏️ Improve Now", type="primary", key="btn_do_improve"):
                    with st.spinner("AI is enhancing your resume writing..."):
                        focus_str = ", ".join(improvement_focus) if improvement_focus else "overall quality"
                        prompt = (
                            f"You are an expert resume writer. Improve the following resume focusing on: {focus_str}.\n\n"
                            f"Guidelines:\n"
                            f"1. Replace weak verbs with powerful action verbs (Led, Architected, Spearheaded, etc.)\n"
                            f"2. Add realistic quantified metrics where possible\n"
                            f"3. Remove filler words and make bullets concise\n"
                            f"4. Keep the same structure (## headings, ### subheadings, - bullets)\n"
                            f"5. Preserve all factual information\n"
                            f"6. Output ONLY the improved resume\n\n"
                            f"Resume:\n{resume_text}"
                        )
                        result = ask_text_ai(prompt)
                        if result:
                            st.session_state["ai_tool_result"] = result
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Improved Resume")
                    st.markdown(st.session_state["ai_tool_result"])

                    col_imp1, col_imp2 = st.columns(2)
                    with col_imp1:
                        if st.button("✅ Apply Changes to My Resume", type="primary", key="btn_apply_improve"):
                            st.session_state["resume_text"] = st.session_state["ai_tool_result"]
                            st.session_state["resume_log"].append("AI Writing Improvement applied")
                            st.success("🎉 Improved version applied to your resume!")
                    with col_imp2:
                        st.download_button(
                            "📥 Download Improved Version",
                            st.session_state["ai_tool_result"].encode("utf-8"),
                            file_name="Resume_Improved.txt", mime="text/plain"
                        )

            # ---- DRAFT COVER LETTER ----
            elif active_tool == "cover_letter":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>📝 Draft Cover Letter</h3>
                    <p>Generate a tailored cover letter that complements your resume</p>
                </div>
                """, unsafe_allow_html=True)

                job_desc = st.text_area(
                    "Paste the target job description (optional but recommended):",
                    height=120, key="cl_job_desc",
                    placeholder="Paste the job posting here for a highly tailored cover letter..."
                )
                company_name = st.text_input("Company name:", key="cl_company", placeholder="e.g. Google, Microsoft, Infosys")
                tone = st.selectbox("Tone:", ["Professional & Confident", "Enthusiastic & Passionate", "Formal & Corporate", "Friendly & Approachable"], key="cl_tone")

                if st.button("📝 Generate Cover Letter", type="primary", key="btn_do_cover"):
                    with st.spinner("Crafting your personalized cover letter..."):
                        name = header.get("name", "Candidate")
                        title = header.get("title", "")
                        prompt = (
                            f"You are a career coach and professional cover letter writer.\n"
                            f"Write a compelling, {tone.lower()} cover letter for {name}"
                            f"{f', applying for {title}' if title else ''}"
                            f"{f' at {company_name}' if company_name.strip() else ''}.\n\n"
                            f"Base it on this resume:\n{resume_text[:4000]}\n\n"
                            f"{f'Target Job Description:{chr(10)}{job_desc[:2000]}' if job_desc.strip() else ''}\n\n"
                            f"Guidelines:\n"
                            f"1. Write 3-4 paragraphs: hook, value proposition, specific achievements, closing\n"
                            f"2. Reference specific skills and accomplishments from the resume\n"
                            f"3. If a job description is provided, align closely with its requirements\n"
                            f"4. Be authentic, avoid generic phrases\n"
                            f"5. Include a professional salutation and sign-off\n"
                            f"6. Output ONLY the cover letter text"
                        )
                        result = ask_text_ai(prompt)
                        if result:
                            st.session_state["ai_tool_result"] = result
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Your Cover Letter")
                    st.markdown(st.session_state["ai_tool_result"])
                    safe_name = (header.get("name") or "Cover_Letter").replace(" ", "_")
                    st.download_button(
                        "📥 Download Cover Letter",
                        st.session_state["ai_tool_result"].encode("utf-8"),
                        file_name=f"Cover_Letter_{safe_name}.txt",
                        mime="text/plain", type="primary"
                    )

            # ---- GENERATE SUMMARY ----
            elif active_tool == "summary":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>📋 Generate Professional Summary</h3>
                    <p>Create a powerful executive summary from your resume content</p>
                </div>
                """, unsafe_allow_html=True)

                summary_style = st.selectbox(
                    "Summary style:",
                    ["Executive Summary (3-4 sentences)", "Elevator Pitch (2 sentences)", "Detailed Profile (5-6 sentences)", "Objective Statement"],
                    key="summary_style"
                )

                if st.button("📋 Generate Summary", type="primary", key="btn_do_summary"):
                    with st.spinner("Crafting your professional summary..."):
                        prompt = (
                            f"You are an executive resume writer. Generate a {summary_style.lower()} "
                            f"based on this resume:\n\n{resume_text[:5000]}\n\n"
                            f"Guidelines:\n"
                            f"1. Highlight the candidate's strongest selling points\n"
                            f"2. Include years of experience, key technologies, and leadership qualities\n"
                            f"3. Use powerful language — no generic phrases\n"
                            f"4. Make it ATS-friendly with relevant keywords\n"
                            f"5. Output ONLY the summary paragraph(s), nothing else"
                        )
                        result = ask_text_ai(prompt)
                        if result:
                            st.session_state["ai_tool_result"] = result
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Your Professional Summary")
                    st.markdown(f'<div class="ai-result-box">{st.session_state["ai_tool_result"]}</div>', unsafe_allow_html=True)
                    if st.button("📋 Copy to Clipboard", key="btn_copy_summary"):
                        st.code(st.session_state["ai_tool_result"], language=None)
                        st.info("Select and copy the text above.")

            # ---- INTERVIEW PREP ----
            elif active_tool == "interview_prep":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>🎤 Interview Preparation</h3>
                    <p>Get tailored interview questions with expert answers for your target role</p>
                </div>
                """, unsafe_allow_html=True)

                role_for_interview = st.text_input(
                    "Target role for interview:",
                    value=header.get("title", ""),
                    key="interview_role_input",
                    placeholder="e.g. Senior Python Developer"
                )
                interview_type = st.selectbox(
                    "Interview type:",
                    ["Technical + Behavioral Mix", "Behavioral Only", "Technical Only", "System Design", "Case Study"],
                    key="interview_type"
                )
                num_questions = st.slider("Number of questions:", 5, 15, 8, key="interview_num_q")

                if st.button("🎤 Generate Questions", type="primary", key="btn_do_interview"):
                    with st.spinner("Preparing interview questions and answers..."):
                        prompt = (
                            f"You are a senior technical interviewer at a top tech company.\n"
                            f"Generate {num_questions} {interview_type.lower()} interview questions for a "
                            f"{role_for_interview} candidate.\n\n"
                            f"Base questions on this resume:\n{resume_text[:4000]}\n\n"
                            f"Format each question as:\n"
                            f"### Q1: [Question]\n"
                            f"**Best Answer:** [A detailed, impressive answer the candidate should give based on their resume]\n\n"
                            f"**Tips:** [1-2 line interview tip]\n\n---\n\n"
                            f"Make questions specific to the candidate's experience. Output ONLY the Q&A."
                        )
                        result = ask_text_ai(prompt)
                        if result:
                            st.session_state["ai_tool_result"] = result
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Interview Preparation Guide")
                    st.markdown(st.session_state["ai_tool_result"])
                    safe_name = (header.get("name") or "Interview_Prep").replace(" ", "_")
                    st.download_button(
                        "📥 Download Interview Guide",
                        st.session_state["ai_tool_result"].encode("utf-8"),
                        file_name=f"Interview_Prep_{safe_name}.txt",
                        mime="text/plain", type="primary"
                    )

            # ---- LINKEDIN OPTIMIZER ----
            elif active_tool == "linkedin":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>💼 LinkedIn Profile Optimizer</h3>
                    <p>Generate an optimized LinkedIn headline, about section, and experience bullets</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button("💼 Optimize for LinkedIn", type="primary", key="btn_do_linkedin"):
                    with st.spinner("Optimizing your LinkedIn profile content..."):
                        prompt = (
                            f"You are a LinkedIn profile optimization expert and personal branding coach.\n"
                            f"Based on this resume, generate optimized LinkedIn profile content:\n\n"
                            f"{resume_text[:5000]}\n\n"
                            f"Generate these sections:\n\n"
                            f"## LinkedIn Headline\n"
                            f"(1 line, max 220 chars, keyword-rich, compelling)\n\n"
                            f"## About Section\n"
                            f"(3-4 paragraphs, storytelling approach, first person, include call-to-action)\n\n"
                            f"## Experience Bullets\n"
                            f"(Rewritten for LinkedIn style - more conversational than resume, still impactful)\n\n"
                            f"## Featured Skills\n"
                            f"(Top 10 skills to endorse, ordered by relevance)\n\n"
                            f"Output ONLY the LinkedIn content."
                        )
                        result = ask_text_ai(prompt)
                        if result:
                            st.session_state["ai_tool_result"] = result
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Your Optimized LinkedIn Content")
                    st.markdown(st.session_state["ai_tool_result"])
                    st.download_button(
                        "📥 Download LinkedIn Content",
                        st.session_state["ai_tool_result"].encode("utf-8"),
                        file_name="LinkedIn_Optimized.txt",
                        mime="text/plain", type="primary"
                    )

            # ---- SKILL GAP ANALYZER ----
            elif active_tool == "skill_gap":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>📊 Skill Gap Analyzer</h3>
                    <p>Compare your skills against a job description to discover gaps and opportunities</p>
                </div>
                """, unsafe_allow_html=True)

                job_desc_gap = st.text_area(
                    "Paste the target job description:",
                    height=150, key="skill_gap_jd_input",
                    placeholder="Paste the full job posting here..."
                )

                if st.button("📊 Analyze Gaps", type="primary", key="btn_do_skill_gap"):
                    if not job_desc_gap.strip():
                        st.warning("Please paste a job description to analyze against.")
                    else:
                        with st.spinner("Analyzing skill gaps..."):
                            prompt = (
                                f"You are a career advisor and skills assessment expert.\n"
                                f"Compare this candidate's resume against the job description and provide a detailed skill gap analysis.\n\n"
                                f"Resume:\n{resume_text[:4000]}\n\n"
                                f"Job Description:\n{job_desc_gap[:3000]}\n\n"
                                f"Provide this analysis:\n\n"
                                f"## Match Score: [percentage]%\n\n"
                                f"## ✅ Skills You Have (Matched)\n"
                                f"- List each matching skill with how it maps to the JD\n\n"
                                f"## ❌ Skills You're Missing\n"
                                f"- List each missing skill and why it matters\n\n"
                                f"## 📈 Recommended Learning Path\n"
                                f"- Specific courses, certifications, or projects to close each gap\n"
                                f"- Estimated time to acquire each skill\n\n"
                                f"## 💡 Quick Wins\n"
                                f"- Skills you partially have that can be quickly strengthened\n\n"
                                f"Output ONLY the analysis."
                            )
                            result = ask_text_ai(prompt)
                            if result:
                                st.session_state["ai_tool_result"] = result
                                st.rerun()
                            else:
                                st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Skill Gap Analysis")
                    st.markdown(st.session_state["ai_tool_result"])

            # ---- EMAIL FOLLOW-UP ----
            elif active_tool == "email_followup":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>📧 Email Follow-Up Generator</h3>
                    <p>Draft a professional follow-up email after applying or interviewing</p>
                </div>
                """, unsafe_allow_html=True)

                email_type = st.selectbox(
                    "Follow-up type:",
                    ["After Submitting Application", "After Phone Screen", "After Technical Interview",
                     "After Final Interview", "Thank You Note", "Checking Application Status"],
                    key="email_type"
                )
                company_email = st.text_input("Company name:", key="email_company", placeholder="e.g. Google")
                interviewer_name = st.text_input("Interviewer/Recruiter name (optional):", key="email_interviewer", placeholder="e.g. Sarah Johnson")

                if st.button("📧 Generate Email", type="primary", key="btn_do_email"):
                    with st.spinner("Drafting your follow-up email..."):
                        name = header.get("name", "Candidate")
                        title = header.get("title", "")
                        prompt = (
                            f"You are a professional career coach. Draft a {email_type.lower()} email for {name}"
                            f"{f' applying for {title}' if title else ''}"
                            f"{f' at {company_email}' if company_email.strip() else ''}.\n"
                            f"{f'Address it to {interviewer_name}.' if interviewer_name.strip() else ''}\n\n"
                            f"Resume context:\n{resume_text[:2000]}\n\n"
                            f"Guidelines:\n"
                            f"1. Professional yet warm tone\n"
                            f"2. Reference specific points from the resume relevant to the role\n"
                            f"3. Keep it concise (under 200 words)\n"
                            f"4. Include subject line\n"
                            f"5. Output ONLY the email"
                        )
                        result = ask_text_ai(prompt)
                        if result:
                            st.session_state["ai_tool_result"] = result
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Your Follow-Up Email")
                    st.markdown(f'<div class="ai-result-box">{st.session_state["ai_tool_result"]}</div>', unsafe_allow_html=True)
                    st.download_button(
                        "📥 Download Email",
                        st.session_state["ai_tool_result"].encode("utf-8"),
                        file_name="Follow_Up_Email.txt",
                        mime="text/plain", type="primary"
                    )

            # ---- TAILOR RESUME ----
            elif active_tool == "tailor_resume":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>🎯 Tailor Resume to Job</h3>
                    <p>Automatically customize your resume keywords and bullets for a specific posting</p>
                </div>
                """, unsafe_allow_html=True)

                job_desc_tailor = st.text_area(
                    "Paste the target job description:",
                    height=150, key="tailor_jd_input",
                    placeholder="Paste the full job posting you want to tailor your resume for..."
                )

                if st.button("🎯 Tailor My Resume", type="primary", key="btn_do_tailor"):
                    if not job_desc_tailor.strip():
                        st.warning("Please paste a job description to tailor against.")
                    else:
                        with st.spinner("Tailoring your resume to the job description..."):
                            prompt = (
                                f"You are an expert resume tailor and ATS optimizer.\n"
                                f"Rewrite this resume to perfectly align with the target job description.\n\n"
                                f"Original Resume:\n{resume_text}\n\n"
                                f"Target Job Description:\n{job_desc_tailor[:3000]}\n\n"
                                f"Guidelines:\n"
                                f"1. Mirror keywords and phrases from the job description naturally\n"
                                f"2. Reorder skills to prioritize those mentioned in the JD\n"
                                f"3. Adjust experience bullets to emphasize relevant achievements\n"
                                f"4. Keep all facts accurate — do not fabricate\n"
                                f"5. Maintain the same format (## headings, ### subheadings, - bullets)\n"
                                f"6. Output ONLY the tailored resume"
                            )
                            result = ask_text_ai(prompt)
                            if result:
                                st.session_state["ai_tool_result"] = result
                                st.rerun()
                            else:
                                st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Tailored Resume")
                    st.markdown(st.session_state["ai_tool_result"])

                    col_t1, col_t2 = st.columns(2)
                    with col_t1:
                        if st.button("✅ Apply Tailored Version", type="primary", key="btn_apply_tailor"):
                            st.session_state["resume_text"] = st.session_state["ai_tool_result"]
                            st.session_state["resume_log"].append("Tailored to specific job")
                            st.success("🎉 Tailored version applied to your resume!")
                    with col_t2:
                        st.download_button(
                            "📥 Download Tailored Resume",
                            st.session_state["ai_tool_result"].encode("utf-8"),
                            file_name="Resume_Tailored.txt", mime="text/plain"
                        )

            # ---- GRAMMAR CHECK ----
            elif active_tool == "grammar_check":
                st.markdown("""
                <div class="tool-panel-header">
                    <h3>✅ Grammar & Style Check</h3>
                    <p>Find and fix grammar errors, typos, and awkward phrasing</p>
                </div>
                """, unsafe_allow_html=True)

                if st.button("✅ Run Grammar Check", type="primary", key="btn_do_grammar"):
                    with st.spinner("Scanning for grammar and style issues..."):
                        prompt = (
                            f"You are an expert proofreader and resume writing specialist.\n"
                            f"Analyze this resume for grammar, spelling, punctuation, and style issues.\n\n"
                            f"Resume:\n{resume_text}\n\n"
                            f"Provide your report in this format:\n\n"
                            f"## 📊 Overall Score: [X/10]\n\n"
                            f"## ❌ Issues Found\n"
                            f"For each issue:\n"
                            f"- **Line:** \"[original text]\"\n"
                            f"- **Issue:** [description]\n"
                            f"- **Fix:** \"[corrected text]\"\n\n"
                            f"## ✅ Corrected Resume\n"
                            f"[Output the full corrected resume with all fixes applied, maintaining the same format]\n\n"
                            f"Output ONLY the report."
                        )
                        result = ask_text_ai(prompt)
                        if result:
                            st.session_state["ai_tool_result"] = result
                            st.rerun()
                        else:
                            st.error(BUSY_MSG)

                if st.session_state.get("ai_tool_result"):
                    st.markdown("### ✅ Grammar & Style Report")
                    st.markdown(st.session_state["ai_tool_result"])

# ============================================================
# ---- 13. PAGE: MY RESUMES (saved to Supabase) ----
# ============================================================
elif st.session_state["page"] == "my_resumes":
    _mr_user = sb_current_user()
    st.markdown('<div class="section-head">📚 My Resumes</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Resumes you saved to your account. Load one to keep editing, or delete it.</div>', unsafe_allow_html=True)

    if not sb_enabled():
        st.info("Cloud storage isn't configured yet. Add SUPABASE_URL and SUPABASE_ANON_KEY to enable saving.")
    elif _mr_user is None:
        st.warning("Please sign in to view your saved resumes.")
        if st.button("Sign in", type="primary"):
            _signin_dialog()
    else:
        try:
            _saved = sb_list_resumes(_mr_user.id)
        except Exception as e:
            _saved = []
            st.error(f"Could not load your resumes: {e}")
        if not _saved:
            st.info("You haven't saved any resumes yet. Build one and press **💾 Save**.")
        for row in _saved:
            _c1, _c2, _c3 = st.columns([5, 1.2, 1.2])
            _created = (row.get("created_at") or "")[:10]
            with _c1:
                st.markdown(f"**{row.get('name','Untitled')}**  \n"
                            f"<span style='color:#6b7280;font-size:12px'>Saved {_created}</span>",
                            unsafe_allow_html=True)
            if _c2.button("Load", key=f"load_{row['id']}", use_container_width=True):
                st.session_state["resume_text"] = row.get("resume_text", "")
                try:
                    st.session_state["resume_header"] = json.loads(row.get("header_json") or "{}")
                except Exception:
                    st.session_state["resume_header"] = {}
                st.session_state["chosen_template_id"] = row.get("template_id")
                st.session_state["page"] = "builder"
                st.rerun()
            if _c3.button("Delete", key=f"del_{row['id']}", use_container_width=True):
                try:
                    sb_delete_resume(_mr_user.id, row["id"])
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not delete: {e}")
            st.markdown("---")
