from flask import Flask, request, jsonify, render_template
import os
import json
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import google.generativeai as genai

app = Flask(__name__)

# ---------- Gemini config ----------
# Reads API key from environment variable GEMINI_API_KEY
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# ---------- 0. Paths ----------
FAQ_PATH = os.path.join("data", "faq.json")
FEES_PATH = os.path.join("data", "fees.json")

# ---------- 1. Load FAQ dataset ----------
faq_data = []
if os.path.exists(FAQ_PATH):
    with open(FAQ_PATH, "r", encoding="utf-8") as f:
        faq_data = json.load(f)
    print(f"Loaded {len(faq_data)} FAQ items.")
else:
    print("⚠ No FAQ dataset found, FAQ support will be limited.")

# ---------- 2. Load Fees dataset ----------
fees_data = {"UG": [], "PG": []}
if os.path.exists(FEES_PATH):
    with open(FEES_PATH, "r", encoding="utf-8") as f:
        fees_data = json.load(f)
    print(
        f"Loaded {len(fees_data.get('UG', []))} UG fee rows and "
        f"{len(fees_data.get('PG', []))} PG fee rows."
    )
else:
    print("⚠ No fees.json found. Fee lookup will be disabled.")

# ---------- 2.5 LangChain vector store ----------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = None
if os.path.exists("vector_store"):
    vectordb = FAISS.load_local(
        "vector_store",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ Vector store loaded.")
else:
    print("⚠ No vector store found. Run build_index.py first.")

# ---------- 3. Link map ----------
LINK_MAP = {
    # Admissions / applications
    "admission": "https://annaadarsh.edu.in/admission-form/",
    "admissions": "https://annaadarsh.edu.in/admission-form/",
    "application": "https://annaadarsh.edu.in/admission-form/",
    "apply": "https://annaadarsh.edu.in/admission-form/",

    # Courses
    "courses": "https://annaadarsh.edu.in/services/under-graduate-courses/",
    "ug courses": "https://annaadarsh.edu.in/services/under-graduate-courses/",
    "under graduate": "https://annaadarsh.edu.in/services/under-graduate-courses/",
    "undergraduate courses": "https://annaadarsh.edu.in/services/under-graduate-courses/",
    "pg courses": "https://annaadarsh.edu.in/services/post-graduate-courses/",
    "post graduate": "https://annaadarsh.edu.in/services/post-graduate-courses/",
    "postgraduate courses": "https://annaadarsh.edu.in/services/post-graduate-courses/",

    # FAQ
    "faq": "https://annaadarsh.edu.in/frequently-asked-questions/",
    "frequently asked questions": "https://annaadarsh.edu.in/frequently-asked-questions/",

    # Fees PDF
    "fees": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",
    "fee": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",
    "fee structure": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",
    "fees structure": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",
    "tuition fees": "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf",

    # Faculty details
    "faculty": "https://annaadarsh.edu.in/faculty-details/#",
    "faculty details": "https://annaadarsh.edu.in/faculty-details/#",
    "teachers": "https://annaadarsh.edu.in/faculty-details/#",
    "staff": "https://annaadarsh.edu.in/faculty-details/#",

    # IQAC
    "iqac": "https://annaadarsh.edu.in/about-iqac-2/#",
    "internal quality": "https://annaadarsh.edu.in/about-iqac-2/#",
    "quality assurance": "https://annaadarsh.edu.in/about-iqac-2/#",
    "about iqac": "https://annaadarsh.edu.in/about-iqac-2/#",

    # Infrastructure / facilities
    "infrastructure": "https://annaadarsh.edu.in/study-infrastructure/",
    "labs": "https://annaadarsh.edu.in/study-infrastructure/",
    "laboratory": "https://annaadarsh.edu.in/study-infrastructure/",
    "library": "https://annaadarsh.edu.in/study-infrastructure/",
    "facilities": "https://annaadarsh.edu.in/study-infrastructure/",
    "campus facilities": "https://annaadarsh.edu.in/study-infrastructure/",
    "study infrastructure": "https://annaadarsh.edu.in/study-infrastructure/",

    # RTI
    "rti": "https://annaadarsh.edu.in/wp-content/uploads/2022/09/AACW-RTI.pdf",
    "right to information": "https://annaadarsh.edu.in/wp-content/uploads/2022/09/AACW-RTI.pdf",
    "rti pdf": "https://annaadarsh.edu.in/wp-content/uploads/2022/09/AACW-RTI.pdf",

    # External college info (EasyCollege)
    "easy college": "https://easycollege.in/annaadarsh/college/",
    "easycollege": "https://easycollege.in/annaadarsh/college/",
    "college overview": "https://easycollege.in/annaadarsh/college/",
    "about college": "https://easycollege.in/annaadarsh/college/",
    "ratings": "https://easycollege.in/annaadarsh/college/",
    "reviews": "https://easycollege.in/annaadarsh/college/",

    # Contact page
    "contact": "https://annaadarsh.edu.in/contacts/",
    "contact us": "https://annaadarsh.edu.in/contacts/",
    "contact page": "https://annaadarsh.edu.in/contacts/",
    "phone number": "https://annaadarsh.edu.in/contacts/",
    "email": "https://annaadarsh.edu.in/contacts/",
    "admission contact": "https://annaadarsh.edu.in/contacts/",

    # General site
    "college website": "https://annaadarsh.edu.in/",
    "official website": "https://annaadarsh.edu.in/",
    "anna adarsh": "https://annaadarsh.edu.in/",
}


def get_link_reply(cleaned_query: str):
    for keyword, url in LINK_MAP.items():
        if keyword in cleaned_query:
            return f"You can check the official details here:\n{url}"
    return None


# ---------- 4. Fee helper functions ----------
STOP_WORDS = {
    "fee", "fees", "course", "courses", "structure", "details",
    "for", "of", "the", "ug", "pg", "college"
}


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def match_course_multiple(query: str, level: str):
    q = normalize(query)
    tokens = [t for t in re.split(r"\W+", q) if t and t not in STOP_WORDS]

    scored_rows = []
    best_score = 0

    for row in fees_data.get(level, []):
        haystack = " ".join([row.get("name", "")] + row.get("keywords", [])).lower()
        score = sum(1 for tok in tokens if tok in haystack)
        if score > 0:
            scored_rows.append((row, score))
            best_score = max(best_score, score)

    if best_score == 0:
        return []

    best_rows = [row for row, score in scored_rows if score == best_score]
    best_rows.sort(key=lambda r: r.get("shift", ""))
    return best_rows


def format_single_course(row, level_label: str):
    y1, y2, y3 = row.get("year1"), row.get("year2"), row.get("year3")

    lines = [
        f"**{row['name']}** ({level_label}, Shift: {row.get('shift', '-')})",
    ]
    if y1 is not None:
        lines.append(f"- 1st Year: ₹{y1}")
    if y2 is not None:
        lines.append(f"- 2nd Year: ₹{y2}")
    if y3 is not None:
        lines.append(f"- 3rd Year: ₹{y3}")

    lines.append(
        "\nFor full official details, please refer to the fee PDF:\n"
        "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf"
    )
    return "\n".join(lines)


def format_course_list(level: str):
    label = "Undergraduate (UG)" if level == "UG" else "Postgraduate (PG)"
    rows = fees_data.get(level, [])

    if not rows:
        return f"No {label} fee data available."

    lines = [f"**{label} Courses & Fees**:\n"]

    for row in rows:
        y1, y2, y3 = row.get("year1"), row.get("year2"), row.get("year3")
        block = [f"• **{row['name']}** (Shift: {row.get('shift', '-')})"]
        if y1 is not None:
            block.append(f"  - 1st Year: ₹{y1}")
        if y2 is not None:
            block.append(f"  - 2nd Year: ₹{y2}")
        if y3 is not None:
            block.append(f"  - 3rd Year: ₹{y3}")
        lines.append("\n".join(block))
        lines.append("")

    lines.append(
        "For full official details, please refer to the fee PDF:\n"
        "https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf"
    )
    return "\n".join(lines)


# ---------- 4.6 College-related keyword set ----------
COLLEGE_KEYWORDS = {
    "anna", "adarsh", "college", "campus",
    "course", "courses", "department", "departments",
    "fee", "fees", "tuition",
    "admission", "admissions", "apply",
    "hostel", "infrastructure", "library",
    "ug", "pg", "shift", "faculty", "placement", "placements",
    "facility", "facilities", "canteen", "sports",
    "lab", "labs", "classroom", "classrooms",
    "environment", "college life", "student life", "campus life"
}

# ---------- 4.7 Make URLs clickable ----------
URL_PATTERN = re.compile(r'(https?://[^\s<>"]+)')


def make_links_clickable(text: str) -> str:
    return URL_PATTERN.sub(
        r'<a href="\1" target="_blank" rel="noopener noreferrer">\1</a>',
        text
    )


# ---------- 4.8 Default college summary ----------
def default_college_summary() -> str:
    return (
        "<strong>Anna Adarsh College for Women</strong> is a reputed institution "
        "located in Anna Nagar, Chennai, Tamil Nadu.<br><br>"
        "<u><strong>About the college:</strong></u><br>"
        "• Established and managed by the D.K. Education Committee.<br>"
        "• Offers a wide range of UG and PG programmes in Arts, Science and Commerce.<br>"
        "• NAAC-accredited, with strong focus on academics and student support.<br>"
        "• Well-equipped laboratories, digital library, seminar halls and smart classrooms.<br>"
        "• Active placement cell offering training, internships and campus recruitment.<br>"
        "• Facilities include hostel, canteen, sports and various student clubs.<br><br>"
        "You can ask me about <strong>courses</strong>, <strong>fees</strong>, "
        "<strong>admissions</strong>, <strong>hostel</strong> or "
        "<strong>infrastructure</strong> for more details."
    )


# ---------- Gemini helper ----------
def answer_with_gemini(user_query: str) -> str:
    """Use Google's Gemini model to answer general (non-college) questions."""
    try:
        model = genai.GenerativeModel("gemini-flash-latest")
        response = model.generate_content(user_query)

        text = getattr(response, "text", None)
        if not text and getattr(response, "candidates", None):
            parts = []
            for c in response.candidates:
                for p in c.content.parts:
                    if hasattr(p, "text"):
                        parts.append(p.text)
            text = "\n".join(parts)

        if not text:
            return "I’m not sure how to answer that right now, sorry."

        return text.strip()
    except Exception as e:
        print("Gemini error TYPE:", type(e).__name__)
        print("Gemini error DETAILS:", e)
        return (
            "The AI service is temporarily unavailable. "
            "Please try again soon, or ask about admissions, fees, or courses."
        )


# ---------- 5. Main answer logic ----------
def get_best_answer(user_query, top_k=3, threshold=0.2):
    cleaned_query = user_query.lower().strip()

    # --- Small-talk / greeting shortcuts ---
    if cleaned_query in ["hi", "hii", "hey", "hello", "hello!", "hi!", "hey!"]:
        return "Hello! How can I help you today?"
    if cleaned_query in ["good morning", "good afternoon", "good evening"]:
        return "Hi there! What can I do for you?"

    # --- "What can you do?" / capabilities question ---
    if cleaned_query in [
        "what can you do",
        "what can u do",
        "what all can you do",
        "what all can u do",
        "what are your features",
        "how can you help me",
        "what can this chatbot do",
        "what can you help me with",
    ]:
        return (
            "I’m your Anna Adarsh assistant! Here’s what I can do:\n\n"
            "• Answer questions about the college, courses, timings, facilities, and contact details.\n"
            "• Show UG and PG fee details for different courses.\n"
            "• Help with common admission- and campus-related doubts.\n"
            "• For general topics (like AI, programming, etc.), I use an AI model (Gemini) to explain things.\n\n"
            "You can try asking things like:\n"
            "• \"UG fees\"\n"
            "• \"B.Sc Computer Science fees\"\n"
            "• \"College address\"\n"
            "• \"What is AI?\""
        )
        # --- High-level "tell me about this college" ---
    if any(
        phrase in cleaned_query
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
        # Always return our curated college summary
        return default_college_summary()


    # --- "Who made you?" / "Who created you?" ---
    if any(
        phrase in cleaned_query
        for phrase in [
            "who made you",
            "who created you",
            "who developed you",
            "who built you",
            "who designed you",
        ]
    ):
        return (
            "This chatbot was developed by <strong>Ms. Shirlyn Danita</strong> "
            "as a project for <strong>Anna Adarsh College for Women</strong>.\n\n"
            "It uses a combination of:\n"
            "• Custom data about the college (courses, fees, facilities, etc.)\n"
            "• A question-answer system built with Python and Flask\n"
            "• Google’s Gemini model to help answer general questions."
        )

    # --- Vague follow-ups like "anything else" / "tell me more" ---
    if cleaned_query in [
        "anything else",
        "anything more",
        "tell me more",
        "tell me more about this college",
        "more about this college",
        "more details",
        "more details about this college",
        "say more about this college",
    ]:
        # Always treat as "tell me more about Anna Adarsh College"
        return default_college_summary()

    # --- "Direct link" shortcuts ---
    if "direct link" in cleaned_query or "direct lin" in cleaned_query:
        return (
            "Here are some useful direct links for Anna Adarsh College:\n\n"
            "• Admission form: https://annaadarsh.edu.in/admission-form/\n"
            "• UG / PG fee PDF: https://annaadarsh.edu.in/wp-content/uploads/2025/03/Fees-for-25-26-website.pdf\n"
            "• Contact page: https://annaadarsh.edu.in/contacts/\n"
            "• Official website: https://annaadarsh.edu.in/\n\n"
            "You can also ask things like \"UG fees\", \"PG fees\" or "
            "\"B.Sc Computer Science fees\" for more specific details."
        )

    # --- Quick fee follow-ups ---
    if cleaned_query in ["ug", "ug fee", "ug fees", "ug courses"]:
        return format_course_list("UG")
    if cleaned_query in ["pg", "pg fee", "pg fees", "pg courses"]:
        return format_course_list("PG")
    if cleaned_query in ["all", "all fees", "all fee", "complete fee structure", "all courses fees"]:
        return format_course_list("UG") + "\n\n" + format_course_list("PG")

    # --- College address + Google Maps ---
    if any(
        phrase in cleaned_query
        for phrase in [
            "college address",
            "address of the college",
            "anna adarsh address",
            "anna adarsh location",
            "where is the college",
            "college location",
            "map link",
            "google map",
        ]
    ):
        return (
            "📍 <strong>ANNA ADARSH COLLEGE FOR WOMEN</strong><br>"
            "II Street, A-1, 9th Main Rd,<br>"
            "Anna Nagar, Chennai, Tamil Nadu 600040.<br><br>"
            "Google Maps Location:<br>"
            "https://www.google.com/maps/place/Anna+Adarsh+College+for+Women/"
        )

    # --- Contact details ---
    if any(
        phrase in cleaned_query
        for phrase in [
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
        return (
            "Here are the official contact details for Anna Adarsh College:<br><br>"
            "📞 <strong>Phone:</strong> "
            "<a href=\"tel:04426212089\">044-26212089</a><br>"
            "📧 <strong>General Email:</strong> "
            "<a href=\"mailto:aacw@annaadarsh.edu.in\">aacw@annaadarsh.edu.in</a><br>"
            "📧 <strong>Admissions Email:</strong> "
            "<a href=\"mailto:admissions@annaadarsh.edu.in\">admissions@annaadarsh.edu.in</a><br><br>"
            "🌐 <strong>Contact Page:</strong> https://annaadarsh.edu.in/contacts/"
        )

    # ---- Fee logic (smart) ----
    if "fee" in cleaned_query or "fees" in cleaned_query or "tuition" in cleaned_query:
        wants_ug = any(x in cleaned_query for x in ["ug", "under graduate", "undergraduate"])
        wants_pg = any(x in cleaned_query for x in ["pg", "post graduate", "postgraduate"])

        if wants_ug and not wants_pg:
            return format_course_list("UG")
        if wants_pg and not wants_ug:
            return format_course_list("PG")

        ug_matches = match_course_multiple(cleaned_query, "UG")
        pg_matches = match_course_multiple(cleaned_query, "PG")

        if ug_matches or pg_matches:
            out = []
            for row in ug_matches:
                out.append(format_single_course(row, "Undergraduate (UG)"))
                out.append("")
            for row in pg_matches:
                out.append(format_single_course(row, "Postgraduate (PG)"))
                out.append("")
            return "\n".join(out)

        return (
            "Do you want **UG fees**, **PG fees**, or **all courses**?\n\n"
            "Try typing:\n"
            "- `UG fees`\n"
            "- `PG fees`\n"
            "- `all fees`\n"
        )

    # --- Decide if this is *actually* a college question ---
    tokens = re.findall(r"\w+", cleaned_query)
    is_college_question = any(tok in COLLEGE_KEYWORDS for tok in tokens)

    # If it's NOT a college-related query, send straight to Gemini
    if not is_college_question:
        return answer_with_gemini(user_query)

    # ---- Vector-store FAQ (LangChain) + default college fallback ----
    if not vectordb:
        link_reply = get_link_reply(cleaned_query)
        if link_reply:
            return link_reply
        return default_college_summary()

    docs_scores = vectordb.similarity_search_with_score(cleaned_query, k=top_k)

    if not docs_scores:
        link_reply = get_link_reply(cleaned_query)
        if link_reply:
            return link_reply
        return default_college_summary()

    best_doc, best_score = docs_scores[0]

    # FAISS score is a distance: lower = better
    if best_score > 0.80:
        return default_college_summary()

    answer = best_doc.metadata.get("answer") or best_doc.page_content

    link_reply = get_link_reply(cleaned_query)
    if link_reply:
        return answer + "\n\n" + link_reply
    return answer


# ---------- 6. Routes ----------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = data.get("message", "")

    if not user_message.strip():
        return jsonify({"reply": "Please type something."})

    reply = get_best_answer(user_message)
    reply = make_links_clickable(reply)
    reply_html = reply.replace("\n", "<br>")

    return jsonify({"reply": reply_html})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, debug=True)
