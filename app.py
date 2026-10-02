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
    {"name": "Resume Maker",  "desc": "Make a job resume in seconds",       "status": "Step 5", "ready": True},
    {"name": "Reels Maker",   "desc": "Auto video for Instagram",           "status": "Soon",   "ready": False},
    {"name": "YouTube Shorts","desc": "Auto video for YouTube",             "status": "Soon",   "ready": False},
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

else:
    # Placeholder page for every tool (we build them one by one)
    st.markdown(f'<div class="main-title">{page}</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">This tool is not built yet. It comes in the next steps.</div>',
                unsafe_allow_html=True)
