from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    redirect,
    url_for,
    session,
)
import os
import json
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai
from werkzeug.utils import secure_filename
import PyPDF2

app = Flask(__name__)
app.secret_key = "change-this-secret-key"  # 👉 change this for safety

# ---------- Gemini config ----------
# Make sure GEMINI_API_KEY is set in Cloud Run env vars
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
print("Gemini configured. Key present:", bool(os.getenv("GEMINI_API_KEY")))

# ---------- 0. Paths ----------
FAQ_PATH = os.path.join("data", "faq.json")
FEES_PATH = os.path.join("data", "fees.json")
UPLOAD_FOLDER = os.path.join("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ---------- 1. Load FAQ dataset ----------
faq_data = []
if os.path.exists(FAQ_PATH):
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    print(f"Loaded {len(faq_data)} FAQ items.")
else:
    print("⚠ No FAQ dataset found.")

# ---------- 2. Load Fees dataset ----------
fees_data = {"UG": [], "PG": []}
if os.path.exists(FEES_PATH):
    with open(FEES_PATH, "r", encoding="utf-8") as f:
        fees_data = json.load(f)
    print("Loaded UG/PG fee data.")
else:
    print("⚠ No fees.json found.")

# ---------- 3. Embeddings / FAISS ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = None


def build_faq_index():
    """Rebuild FAISS index from faq_data."""
    global vectordb
    if not faq_data:
        vectordb = None
        print("⚠ No FAQ data: index not built.")
        return

    texts = [item.get("question", "") for item in faq_data]
    metadatas = [{"answer": item.get("answer", "")} for item in faq_data]

    vectordb = FAISS.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
    )
    print(f"✅ FAISS index rebuilt with {len(texts)} items.")


# Build index once at start
build_faq_index()

# ---------- 4. Link map ----------
LINK_MAP = {
    # Admissions / applications
    "admission": "https://annaadarsh.edu.in/admission-form/",
    "admissions": "https://annaadarsh.edu.in/admission-form/",
    "apply": "https://annaadarsh.edu.in/admission-form/",
    "application": "https://annaadarsh.edu.in/admission-form/",

    # Courses
    "courses": "https://annaadarsh.edu.in/services/under-graduate-courses/",
    "ug courses": "https://annaadarsh.edu.in/services/under-graduate-courses/",
    "pg courses": "https://annaadarsh.edu.in/services/post-graduate-courses/",
    "post graduate": "https://annaadarsh.edu.in/services/post-graduate-courses/",
    "postgraduate": "https://annaadarsh.edu.in/services/post-graduate-courses/",

    # Fees
    "fee": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",
    "fees": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",
    "fee structure": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",
    "tuition": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",

    # Faculty
    "faculty": "https://annaadarsh.edu.in/faculty-details/#",
    "teachers": "https://annaadarsh.edu.in/faculty-details/#",

    # Infrastructure
    "infrastructure": "https://annaadarsh.edu.in/study-infrastructure/",
    "library": "https://annaadarsh.edu.in/study-infrastructure/",
    "labs": "https://annaadarsh.edu.in/study-infrastructure/",
    "campus facilities": "https://annaadarsh.edu.in/study-infrastructure/",

    # Contact
    "contact": "https://annaadarsh.edu.in/contacts/",
    "phone number": "https://annaadarsh.edu.in/contacts/",
    "email": "https://annaadarsh.edu.in/contacts/",

    # General
    "college website": "https://annaadarsh.edu.in/",
    "official website": "https://annaadarsh.edu.in/",
}


def get_link_reply(cleaned_query: str):
    for key, url in LINK_MAP.items():
        if key in cleaned_query:
            return f"You can check the official details here:\n{url}"
    return None


# ---------- 5. Fee helper logic ----------
STOP_WORDS = {
    "fee",
    "fees",
    "course",
    "courses",
    "structure",
    "for",
    "of",
    "the",
    "ug",
    "pg",
    "college",
}


def normalize(text: str):
    return re.sub(r"\s+", " ", text.lower()).strip()


def match_course_multiple(query, level):
    q = normalize(query)
    tokens = [t for t in re.split(r"\W+", q) if t and t not in STOP_WORDS]

    scored = []
    best_score = 0
    for row in fees_data.get(level, []):
        haystack = " ".join([row.get("name", "")] + row.get("keywords", [])).lower()
        score = sum(tok in haystack for tok in tokens)
        if score > 0:
            scored.append((row, score))
            best_score = max(best_score, score)

    if best_score == 0:
        return []

    best_rows = [row for row, sc in scored if sc == best_score]
    best_rows.sort(key=lambda r: r.get("shift", ""))
    return best_rows


def format_single_course(row, level_label):
    y1, y2, y3 = row.get("year1"), row.get("year2"), row.get("year3")

    parts = [f"**{row['name']}** ({level_label}, Shift: {row.get('shift', '-')})"]
    if y1:
        parts.append(f"- 1st Year: ₹{y1}")
    if y2:
        parts.append(f"- 2nd Year: ₹{y2}")
    if y3:
        parts.append(f"- 3rd Year: ₹{y3}")

    parts.append(
        "\nFull fee PDF:\nhttps://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf"
    )
    return "\n".join(parts)


def format_course_list(level):
    label = "Undergraduate (UG)" if level == "UG" else "Postgraduate (PG)"
    rows = fees_data.get(level, [])

    if not rows:
        return f"No {label} fee data available."

    out = [f"**{label} Courses & Fees**:\n"]
    for row in rows:
        block = [f"• **{row['name']}** (Shift: {row.get('shift', '-')})"]
        if row.get("year1"):
            block.append(f"  - 1st Year: ₹{row['year1']}")
        if row.get("year2"):
            block.append(f"  - 2nd Year: ₹{row['year2']}")
        if row.get("year3"):
            block.append(f"  - 3rd Year: ₹{row['year3']}")
        out.append("\n".join(block) + "\n")

    out.append(
        "Full fee PDF:\nhttps://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf"
    )
    return "\n".join(out)


# ---------- 6. College summary ----------
def default_college_summary():
    return (
        "<strong>Anna Adarsh College for Women</strong> is a reputed institution "
        "located in Anna Nagar, Chennai, Tamil Nadu.<br><br>"
        "<u><strong>About the college:</strong></u><br>"
        "• Offers a wide range of UG and PG programmes in Arts, Science and Commerce.<br>"
        "• NAAC-accredited, with strong focus on academics and student support.<br>"
        "• Well-equipped laboratories, digital library, seminar halls and smart classrooms.<br>"
        "• Active placement cell offering training, internships and campus recruitment.<br>"
        "• Facilities include hostel, canteen, sports and various student clubs.<br><br>"
        "You can ask me about <strong>courses</strong>, <strong>fees</strong>, "
        "<strong>admissions</strong>, <strong>hostel</strong> or "
        "<strong>infrastructure</strong> for more details."
    )


# ---------- 7. Make links clickable ----------
URL_PATTERN = re.compile(r"(https?://[^\s<>\"]+)")


def make_links_clickable(text: str) -> str:
    return URL_PATTERN.sub(
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        text,
    )


# ---------- 8. Gemini fallback ----------
def answer_with_gemini(query):
    """
    Call Gemini, but force it to speak as AACW Chatbot and
    NEVER say things like 'I am a large language model trained by Google'.
    """
    try:
        system_prompt = (
            "You are the AACW Chatbot, a virtual assistant for Anna Adarsh College for Women.\n"
            "Always respond as this college chatbot.\n"
            "If the user asks who or what you are, you should say exactly:\n"
            "\"I am the AACW Chatbot, a virtual assistant for Anna Adarsh College for Women.\"\n"
            "Never say that you are a large language model, language model, AI model, "
            "or that you were trained by Google.\n"
            "For college-related questions, be accurate and concise. "
            "For general questions (like AI, programming, etc.), answer helpfully "
            "but keep the same AACW Chatbot identity."
        )

        # 👉 FIXED: use the correct Flash model
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=system_prompt,
        )

        res = model.generate_content(query)
        text = getattr(res, "text", None)

        if not text and getattr(res, "candidates", None):
            parts = []
            for c in res.candidates:
                for p in c.content.parts:
                    if hasattr(p, "text"):
                        parts.append(p.text)
            text = "\n".join(parts)

        return text.strip() if text else "I'm not sure how to answer that."
    except Exception as e:
        # This prints the REAL reason if something fails (check Cloud Run logs)
        print("Gemini error:", repr(e))
        return "Gemini is currently unavailable."


# ---------- 9. Analytics ----------
analytics = {
    "total_messages": 0,
    "college_questions": 0,
    "fee_queries": 0,
    "faq_hits": 0,
    "gemini_calls": 0,
}

# ---------- 10. College keywords ----------
COLLEGE_KEYWORDS = {
    "anna",
    "adarsh",
    "college",
    "campus",
    "course",
    "courses",
    "department",
    "departments",
    "fee",
    "fees",
    "tuition",
    "admission",
    "admissions",
    "apply",
    "hostel",
    "infrastructure",
    "library",
    "ug",
    "pg",
    "shift",
    "faculty",
    "placement",
    "placements",
    "facility",
    "facilities",
    "canteen",
    "sports",
    "lab",
    "labs",
    "classroom",
    "classrooms",
    "environment",
    "student",
    "students",
}


# ---------- 11. Main bot logic ----------
def get_best_answer(user_query, top_k=3, threshold=0.2):
    q = user_query.lower().strip()

    # --- 1. Greetings ---
    if q in ["hi", "hii", "hey", "hello", "hello!", "hi!", "hey!"]:
        return "Hello! How can I help you today?"
    if q in ["good morning", "good afternoon", "good evening"]:
        return "Hi there! What can I do for you?"
    
    # --- 2. Appreciation / Thanks ---
    if q in ["thank you", "thanks", "thankyou", "thank u", "tnx"]:
        return "You're welcome! I'm happy to help 😊"

    if q in ["good", "great", "nice", "awesome"]:
        return "I'm glad to hear that! Let me know if you need anything else 😊"

    # --- 3. Bot identity: who / what are you / name / are you AI? ---
    if any(
        phrase in q
        for phrase in [
            "who are you",
            "who r you",
            "who are u",
            "who r u",
            "what are you",
            "what are u",
            "what r you",
            "what r u",
            "who is this",
            "who am i chatting with",
            "who am i talking to",
            "who is this bot",
            "are you ai",
            "are you an ai",
            "are you a bot",
            "are you chatbot",
            "are u ai",
            "are u a bot",
            "are u chatbot",
            "what is your name",
            "whats your name",
            "what's your name",
            "your name",
        ]
    ):
        return (
            "I’m the <strong>AACW Chatbot</strong>, a virtual assistant for "
            "<strong>Anna Adarsh College for Women</strong>.<br><br>"
            "I can help you with information about courses, fees, admissions, "
            "hostel, infrastructure and more about the college, and I can also "
            "answer general questions using Google’s <strong>Gemini</strong> model."
        )

    # --- 4. "Who made you?" / "Who created you?" (detailed project credit) ---
    if any(
        phrase in q
        for phrase in [
            "who made you",
            "who made u",
            "who created you",
            "who created u",
            "who built you",
            "who built u",
            "who developed you",
            "who developed u",
            "who designed you",
            "who designed u",
        ]
    ):
        return (
            "This chatbot was developed by <strong>Ms. Shirlyn Danita</strong> "
            "as a project for <strong>Anna Adarsh College for Women</strong>.<br><br>"
            "It uses custom curated data, a Python + Flask backend, and Google’s Gemini model "
            "to answer college-related and general queries."
        )

    # --- 5. "What can you do?" / capabilities ---
    if (
        re.search(r"what\s+can\s+(you|u)\s+do", q)
        or q in [
            "what all can you do",
            "what all can u do",
            "what are your features",
            "how can you help me",
            "what can this chatbot do",
            "what can you help me with",
        ]
    ):
        return (
            "I’m the <strong>AACW Chatbot</strong>, a virtual assistant for "
            "<strong>Anna Adarsh College for Women</strong>.<br><br>"
            "<u><strong>Here’s what I can help you with for this college:</strong></u><br>"
            "• UG & PG course and programme details.<br>"
            "• Fee information and official fee PDFs.<br>"
            "• Admissions, application links and basic eligibility doubts.<br>"
            "• Hostel and infrastructure information (labs, library, campus facilities).<br>"
            "• College address, contact numbers and email IDs.<br><br>"
            "<u><strong>General AI help:</strong></u><br>"
            "Besides college queries, I can also explain general topics (like AI, "
            "programming, projects and more) using Google’s <strong>Gemini</strong> "
            "model when you ask."
        )

    # --- 6. “Tell me more” / “What else” logic ---
    # 6a. Explicitly about the college → curated summary
    if any(
        phrase in q
        for phrase in [
            "tell me more about this college",
            "tell me more about the college",
            "tell me more about anna adarsh",
            "more about this college",
            "more about the college",
            "more about anna adarsh",
            "anything else about this college",
            "anything else about the college",
        ]
    ):
        analytics["college_questions"] += 1
        return default_college_summary()

    # 6b. Generic “tell me more / what else” → Gemini
    if q in [
        "tell me more",
        "anything else",
        "anything more",
        "what else",
        "say more",
        "more details",
        "more",
    ]:
        analytics["gemini_calls"] += 1
        return answer_with_gemini(user_query)

    # --- 7. High-level "about this college" queries ---
    if any(
        phrase in q
        for phrase in [
            "tell me about this college",
            "tell me about anna adarsh college",
            "tell me about anna adarsh",
            "about this college",
            "about anna adarsh college",
            "about anna adarsh",
            "college overview",
            "overview of this college",
            "overview of the college",
        ]
    ):
        analytics["college_questions"] += 1
        return default_college_summary()

    # --- 8. Direct link shortcuts ---
    if "direct link" in q or "direct lin" in q:
        return (
            "Here are some useful direct links for Anna Adarsh College:\n\n"
            "• Admission form: https://annaadarsh.edu.in/admission-form/\n"
            "• UG / PG fee PDF: https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf\n"
            "• Contact page: https://annaadarsh.edu.in/contacts/\n"
            "• Official website: https://annaadarsh.edu.in/\n\n"
            "You can also ask things like \"UG fees\", \"PG fees\" or "
            "\"B.Sc Computer Science fees\" for more specific details."
        )

    # --- 9. Contact details ---
    if any(
        word in q
        for word in [
            "contact",
            "contact number",
            "phone",
            "phone number",
            "call",
            "email",
            "mail",
            "helpline",
            "support",
            "admission contact",
            "admission email",
            "college phone",
            "telephone",
        ]
    ):
        analytics["college_questions"] += 1
        return (
            "Here are the official contact details for Anna Adarsh College:<br><br>"
            "📞 <strong>Phone:</strong> 044-26212089<br>"
            "📧 <strong>Email:</strong> aacw@annaadarsh.edu.in<br><br>"
            "🌐 <strong>Contact Page:</strong> https://annaadarsh.edu.in/contacts/"
        )

    # --- 10. Fee logic ---
    if "fee" in q or "fees" in q or "tuition" in q:
        analytics["fee_queries"] += 1
        analytics["college_questions"] += 1

        wants_ug = any(x in q for x in ["ug", "under graduate", "undergraduate"])
        wants_pg = any(x in q for x in ["pg", "post graduate", "postgraduate"])

        if wants_ug and not wants_pg:
            return format_course_list("UG")
        if wants_pg and not wants_ug:
            return format_course_list("PG")

        ug_match = match_course_multiple(q, "UG")
        pg_match = match_course_multiple(q, "PG")

        if ug_match or pg_match:
            out = []
            for r in ug_match:
                out.append(format_single_course(r, "Undergraduate (UG)"))
                out.append("")
            for r in pg_match:
                out.append(format_single_course(r, "Postgraduate (PG)"))
                out.append("")
            return "\n".join(out)

        return (
            "Do you want **UG fees**, **PG fees**, or fees for a specific course? "
            "You can also type things like \"B.Sc Computer Science fees\"."
        )

    # --- 11. Decide if college-related ---
    tokens = re.findall(r"\w+", q)
    is_college_question = any(tok in COLLEGE_KEYWORDS for tok in tokens)
    if is_college_question:
        analytics["college_questions"] += 1

    # If NOT college related → Gemini
    if not is_college_question:
        analytics["gemini_calls"] += 1
        return answer_with_gemini(user_query)

    # --- 12. Vector-store FAQ (LangChain) + default fallback ---
    if vectordb:
        docs_scores = vectordb.similarity_search_with_score(q, k=top_k)
        if docs_scores:
            best_doc, best_score = docs_scores[0]
            # FAISS: lower score = closer
            if best_score <= 0.80:
                analytics["faq_hits"] += 1
                answer = best_doc.metadata.get("answer") or best_doc.page_content
                link = get_link_reply(q)
                if link:
                    return answer + "\n\n" + link
                return answer

    # Fallback: summary + optional link
    link = get_link_reply(q)
    base = default_college_summary()
    if link:
        return base + "<br><br>" + link
    return base


# ---------- 12. Auth helpers ----------
ADMIN_PIN = os.getenv("AACW_ADMIN_PIN", "2806")  # 👉 set env var in real use


def is_admin_logged_in():
    return session.get("is_admin") is True


# ---------- 13. Routes – main chat ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    analytics["total_messages"] += 1
    data = request.get_json() or {}
    msg = data.get("message", "")
    reply = get_best_answer(msg)
    reply = make_links_clickable(reply)
    return jsonify({"reply": reply.replace("\n", "<br>")})


# ---------- 14. Admin auth (always PIN on /admin) ----------

# /admin and /admin/login → PIN page
@app.route("/admin", methods=["GET", "POST"])
@app.route("/admin/login", methods=["GET", "POST"])
def admin_login():
    error = None

    if request.method == "POST":
        pin = (request.form.get("pin") or "").strip()
        if pin == ADMIN_PIN:
            session["is_admin"] = True
            return redirect(url_for("admin_panel"))
        else:
            error = "Incorrect PIN"

    return render_template("admin_login.html", error=error)


@app.route("/admin/logout")
def admin_logout():
    session.pop("is_admin", None)
    return redirect(url_for("admin_login"))


# ---------- 15. Admin panel ----------
@app.route("/admin/panel")
def admin_panel():
    if not is_admin_logged_in():
        return redirect(url_for("admin_login"))
    return render_template("admin.html", faqs=faq_data, stats=analytics)


@app.route("/admin/add", methods=["POST"])
def admin_add():
    if not is_admin_logged_in():
        return redirect(url_for("admin_login"))

    question = request.form.get("question", "").strip()
    answer = request.form.get("answer", "").strip()

    if question and answer:
        faq_data.append({"question": question, "answer": answer})
        with open(FAQ_PATH, "w", encoding="utf-8") as f:
            json.dump(faq_data, f, ensure_ascii=False, indent=2)
        build_faq_index()

    return redirect(url_for("admin_panel"))


@app.route("/admin/delete/<int:index>", methods=["POST"])
def admin_delete(index):
    if not is_admin_logged_in():
        return redirect(url_for("admin_login"))

    if 0 <= index < len(faq_data):
        faq_data.pop(index)
        with open(FAQ_PATH, "w", encoding="utf-8") as f:
            json.dump(faq_data, f, ensure_ascii=False, indent=2)
        build_faq_index()
    return redirect(url_for("admin_panel"))


@app.route("/admin/edit/<int:index>", methods=["POST"])
def admin_edit(index):
    if not is_admin_logged_in():
        return redirect(url_for("admin_login"))

    if 0 <= index < len(faq_data):
        question = request.form.get("question", "").strip()
        answer = request.form.get("answer", "").strip()
        if question and answer:
            faq_data[index] = {"question": question, "answer": answer}
            with open(FAQ_PATH, "w", encoding="utf-8") as f:
                json.dump(faq_data, f, ensure_ascii=False, indent=2)
            build_faq_index()
    return redirect(url_for("admin_panel"))


# ---------- 16. PDF Upload & auto-extract ----------
def extract_qa_from_text(text: str):
    """
    Very simple heuristic:
    Looks for patterns like:
    Q: .... ?
    A: ....
    """
    qa_pairs = []
    pattern = re.compile(
        r"Q[:\.\-]\s*(.+?\?)\s*A[:\.\-]\s*(.+?)(?=Q[:\.\-]|$)",
        re.DOTALL | re.IGNORECASE,
    )
    for m in pattern.finditer(text):
        q = m.group(1).strip().replace("\n", " ")
        a = m.group(2).strip().replace("\n", " ")
        qa_pairs.append((q, a))
    return qa_pairs


@app.route("/admin/upload_pdf", methods=["POST"])
def admin_upload_pdf():
    if not is_admin_logged_in():
        return redirect(url_for("admin_login"))

    file = request.files.get("pdf_file")
    if not file or file.filename == "":
        return redirect(url_for("admin_panel"))

    filename = secure_filename(file.filename)
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(path)

    # Extract text
    text = ""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text() or ""

    qa_pairs = extract_qa_from_text(text)

    # Add all extracted QAs
    added_count = 0
    for q, a in qa_pairs:
        faq_data.append({"question": q, "answer": a})
        added_count += 1

    if added_count > 0:
        with open(FAQ_PATH, "w", encoding="utf-8") as f:
            json.dump(faq_data, f, ensure_ascii=False, indent=2)
        build_faq_index()
        print(f"📄 Added {added_count} FAQs from PDF.")
    else:
        print("📄 No Q/A pairs detected in uploaded PDF.")

    return redirect(url_for("admin_panel"))


# ---------- 17. Run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
