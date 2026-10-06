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

# ---- The function that talks to Groq's AI (the "brain") ----
def ask_groq(prompt):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "No key found in .env file."
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    data = {"model": "openai/gpt-oss-120b",
            "messages": [{"role": "user", "content": prompt}]}
    r = requests.post(url, headers=headers, json=data, timeout=60)
    if r.status_code != 200:
        return f"AI error {r.status_code}: {r.text[:200]}"
    return r.json()["choices"][0]["message"]["content"]

# ---- The function that asks Google's AI to draw a picture ----
def ask_gemini_image(prompt):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None, "no key"
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-image:generateContent"
    r = requests.post(url, params={"key": api_key},
                      json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=180)
    if r.status_code != 200:
        return None, f"Gemini error {r.status_code}: {r.text[:200]}"
    parts = r.json().get("candidates", [{}])[0].get("content", {}).get("parts", [])
    for part in parts:
        if "inlineData" in part:
            return base64.b64decode(part["inlineData"]["data"]), None
    return None, "No image in the answer"

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

# ---- Turn the resume text into a real PDF file ----
def make_resume_pdf(text):
    from fpdf import FPDF
    # The AI sometimes writes "smart" characters (fancy spaces, dashes, quotes)
    # that the PDF font cannot print. Replace them with normal ones first.
    replacements = {
        "\u00a0": " ", "\u2002": " ", "\u2003": " ", "\u2009": " ",
        "\u2011": "-", "\u2013": "-", "\u2014": "-", "\u2022": "-",
        "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    }
    for smart, normal in replacements.items():
        text = text.replace(smart, normal)
    text = text.encode("latin-1", errors="replace").decode("latin-1")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_margins(15, 15, 15)
    pdf.add_page()
    for line in text.split("\n"):
        line = line.strip().replace("**", "")
        if not line:
            pdf.ln(3)
        elif line.startswith("# "):
            pdf.set_font("Helvetica", "B", 18)
            pdf.multi_cell(0, 9, line[2:].strip(), new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("## "):
            pdf.set_font("Helvetica", "B", 13)
            pdf.ln(2)
            pdf.multi_cell(0, 8, line[3:].strip(), new_x="LMARGIN", new_y="NEXT")
        elif line.startswith("- "):
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, "  -  " + line[2:].strip(), new_x="LMARGIN", new_y="NEXT")
        else:
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(0, 6, line, new_x="LMARGIN", new_y="NEXT")
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
    return None, "All picture services are busy, try again in a minute"

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
    /* Tool cards: real buttons styled as pretty cards (only on Home page grid) */
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button {
        background: linear-gradient(135deg, #1e1b4b, #312e81);
        color: #ffffff;
        border-radius: 16px;
        padding: 22px 24px;
        min-height: 150px;
        border: 1px solid #4338ca;
        width: 100%;
        white-space: normal;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        justify-content: flex-start;
        transition: transform 0.2s, border-color 0.2s;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button:hover {
        transform: scale(1.02);
        border-color: #818cf8;
        color: #ffffff;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button p {
        width: 100%;
        text-align: left;
        margin: 4px 0;
        font-size: 15px;
        color: #c7d2fe;
    }
    div[data-testid="stHorizontalBlock"] div[data-testid="stButton"] button strong {
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
    {"name": "Chat AI",       "desc": "Talk with AI like ChatGPT",          "status": "Ready",  "ready": True},
    {"name": "Image Generator","desc": "Type words, get a picture",         "status": "Ready",  "ready": True},
    {"name": "Text & Articles","desc": "Write articles, news, posts",       "status": "Ready",  "ready": True},
    {"name": "Resume Maker",  "desc": "Make a job resume in seconds",       "status": "Ready", "ready": True},
    {"name": "Reels Maker",   "desc": "AI video or your own pics",           "status": "Ready",   "ready": True},
    {"name": "YouTube Shorts","desc": "AI video or your own pics",           "status": "Ready",   "ready": True},
    {"name": "Video Editing", "desc": "AI helps edit your videos",          "status": "Soon",   "ready": False},
]

# ---- 5. Show the page ----
if page == "Home":
    st.markdown('<div class="main-title">AI Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">One website, all AI tools. Click a tool in the left menu to open it.</div>',
                unsafe_allow_html=True)

    # Grid of 3 cards per row - cards are real buttons that switch pages safely
    cols = st.columns(3)
    for i, tool in enumerate(tools):
        with cols[i % 3]:
            label = f"**{tool['name']}**  \n{tool['desc']}  \n{tool['status']}"
            st.button(label, key=f"card_{tool['name']}",
                      on_click=go_to, args=(tool["name"],), use_container_width=True)

elif page == "Chat AI":
    st.markdown('<div class="main-title">Chat AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Talk with AI. Your key is safely hidden in the .env file.</div>',
                unsafe_allow_html=True)

    # Remember old messages so they don't disappear
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show all old messages
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.write(m["content"])

    # The typing box at the bottom
    prompt = st.chat_input("Type your message here...")

    if prompt:
        # Show what the user typed
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Get the AI answer
        reply = ask_groq(prompt)

        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.write(reply)

elif page == "Image Generator":
    st.markdown('<div class="main-title">Image Generator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Type words, get a picture.</div>',
                unsafe_allow_html=True)

    prompt = st.text_input("What picture do you want? (English works best)")
    if st.button("Generate Image"):
        if not prompt.strip():
            st.warning("Type something first!")
        else:
            with st.spinner("AI is drawing... please wait"):
                # Try each free provider silently, in order, until one works
                img_bytes, err = ask_gemini_image(prompt)
                if err:
                    img_bytes, err = ask_cf_image(prompt)
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
                    st.error("Sorry, the free image services are busy right now. Please try again in a minute.")
                else:
                    st.image(img_bytes, caption=prompt, use_container_width=True)
                    st.download_button("Download Image", img_bytes,
                                       file_name="my_ai_image.png", mime="image/png")

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
                result = ask_groq(instructions)
                st.markdown(result)
                st.download_button("Download as .txt", result.encode("utf-8"),
                                   file_name="my_text.txt", mime="text/plain")

elif page == "Resume Maker":
    st.markdown('<div class="main-title">Resume Maker</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Fill in the boxes and AI will write your resume. '
                'Download it as PDF or TXT.</div>',
                unsafe_allow_html=True)

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

        submitted = st.form_submit_button("Write my resume")

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
                    f"2. Use ## headings and - bullet points.\n"
                    f"3. Make it impressive and professional.\n"
                    f"4. Keep it under one page if possible.\n"
                    f"5. Write the resume text only, no extra comments.\n"
                    f"6. NEVER invent company names, dates, numbers or facts. "
                    f"If a detail is missing, use [Company name] or [Year] as a placeholder.\n"
                    f"7. End the resume with a '## Personal Details' section containing "
                    f"father's name, date of birth, gender, marital status and strong points."
                )
                resume = ask_groq(instructions)

            st.markdown("### Your AI resume")
            st.markdown(resume)

            # Step 3: download the resume as PDF (for job applications) or TXT
            safe_name = name.replace(" ", "_")
            col_a, col_b = st.columns(2)
            with col_a:
                st.download_button("Download as PDF", make_resume_pdf(resume),
                                   file_name=f"Resume_{safe_name}.pdf",
                                   mime="application/pdf")
            with col_b:
                st.download_button("Download as .txt", resume.encode("utf-8"),
                                   file_name=f"Resume_{safe_name}.txt",
                                   mime="text/plain")

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
                    st.session_state["reels_script"] = ask_groq(instructions)
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
                    st.session_state["yt_script"] = ask_groq(instructions)
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

else:
    # Placeholder page for every tool (we build them one by one)
    st.markdown(f'<div class="main-title">{page}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">This tool is not built yet. It comes in the next steps.</div>',
                unsafe_allow_html=True)
