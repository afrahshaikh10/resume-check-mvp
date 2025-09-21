# ------------------------ IMPORTS ------------------------
import streamlit as st
from src.parser import extract_text_from_file
from sentence_transformers import SentenceTransformer, util
import torch
import re
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
from datetime import datetime
import spacy
import openai
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ------------------------ CUSTOM STYLES ------------------------
# ------------------------ DARKER PASTEL THEME ------------------------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background-color: #C7D8E0;  /* soft darker pastel blue-gray */
        color: #333333;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Main title */
    .stTitle {
        color: #4B7AA3;  /* darker pastel blue */
        font-weight: bold;
    }

    /* Sidebar background */
    .css-1d391kg { 
        background-color: #ffffff;
    }

    /* Buttons */
    .stButton>button {
        background-color: #4B7AA3;  /* darker pastel blue */
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        font-weight: bold;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background-color: #3A6280;
        color: white;
    }

    /* DataFrame table headers */
    .stDataFrame th {
        background-color: #9FB8C8;  /* darker pastel header */
        color: #333333;
    }
    .stDataFrame td {
        background-color: #E2EDF4;  /* lighter table rows */
        color: #333333;
    }

    /* AI Feedback box */
    .ai-feedback {
        background-color: #FFE59C;  /* soft pastel yellow */
        border-left: 5px solid #FFD97D;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 14px;
    }

    /* Candidate summary box */
    .candidate-summary {
        background-color: #A3E7C4;  /* soft pastel green */
        border-left: 5px solid #4FD1C5;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 14px;
    }

    /* Top candidates box */
    .top-candidate {
        background-color: #E9A3F7;  /* soft pastel purple-pink */
        border-left: 5px solid #D96CFF;
        padding: 10px 15px;
        border-radius: 8px;
        margin-bottom: 10px;
        font-size: 14px;
        font-weight: bold;
    }

    /* Sliders */
    .stSlider>div>div>div>div>div>input {
        accent-color: #4B7AA3;
    }

    /* Chart container */
    .stPlotlyChart, .stPyplot {
        background-color: #C7D8E0; 
        border-radius: 12px;
        padding: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# ------------------------ CONFIG ------------------------
st.set_page_config(
    page_title="📄 Resume Relevance Checker",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load OpenAI key from Streamlit secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

# ------------------------ STYLE ------------------------
st.markdown("""
<style>
body {background-color: #f5f7fa;}
h1,h2,h3 {color: #1f3b73;}
.stButton>button {background-color: #ff7f50; color:white; border-radius:10px; height:3em; width:100%; font-size:16px;}
.stDownloadButton>button {background-color: #4CAF50; color:white; border-radius:10px; height:3em; font-size:16px;}
</style>
""", unsafe_allow_html=True)

# ------------------------ MODEL LOAD ------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_llm():
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

model = load_model()
nlp = load_spacy()
tokenizer, llm_model = load_llm()

# ------------------------ DATABASE ------------------------
engine = create_engine("sqlite:///candidates.db")

def save_to_db(results, jd_text):
    df = pd.DataFrame(results)
    df["job_description"] = jd_text
    df["timestamp"] = datetime.now()
    df.to_sql("analysis_results", engine, if_exists="append", index=False)

def load_db():
    try:
        df = pd.read_sql("SELECT * FROM analysis_results", engine)
        return df
    except:
        return pd.DataFrame()

# ------------------------ UTILITIES ------------------------
def compute_relevance(resume_text: str, jd_text: str):
    if not resume_text or not jd_text: return 0.0
    emb_resume = model.encode(resume_text, convert_to_tensor=True)
    emb_jd = model.encode(jd_text, convert_to_tensor=True)
    return round(util.cos_sim(emb_resume, emb_jd).item() * 100, 2)

def extract_skills(text: str):
    skills = [
        "python","java","c++","javascript","r","scala","go",
        "sql","machine learning","deep learning","nlp","data science",
        "pandas","numpy","matplotlib","seaborn","scikit-learn",
        "tensorflow","pytorch","keras",
        "django","flask","react","angular","vue","nodejs","express",
        "html","css","bootstrap",
        "git","docker","kubernetes","aws","azure","gcp",
        "tableau","powerbi","excel","spark","hadoop",
        "mysql","postgresql","mongodb","redis","elasticsearch"
    ]
    return set([s for s in skills if re.search(rf"\b{s}\b", text, re.IGNORECASE)])

def get_verdict(score):
    if score > 70: return "High"
    elif score > 40: return "Medium"
    else: return "Low"

def short_summary(text, length=50):
    words = text.split()
    return " ".join(words[:length]) + ("..." if len(words) > length else "")

def extract_entities(text):
    doc = nlp(text)
    entities = {
        "Names": [], "Emails": [], "Phones": [], "Education": [], "Organizations": [], "Skills": extract_skills(text)
    }
    entities["Emails"] = list(set(re.findall(r'\b[\w\.-]+@[\w\.-]+\.\w{2,4}\b', text)))
    entities["Phones"] = list(set(re.findall(r'\+?\d[\d -]{8,12}\d', text)))
    for ent in doc.ents:
        if ent.label_=="PERSON": entities["Names"].append(ent.text)
        elif ent.label_=="ORG": entities["Organizations"].append(ent.text)
        elif ent.label_ in ["EDUCATION","DEGREE"]: entities["Education"].append(ent.text)
    for k in entities: entities[k] = list(set(entities[k]))
    return entities

# ------------------------ AI FEEDBACK ------------------------
def generate_ai_feedback(resume_text, jd_text, found_skills, missing_skills, max_words=450):
    resume_short = " ".join(resume_text.split()[:max_words])
    prompt = f"""
You are an HR assistant. Job description:

{jd_text}

Candidate has these skills: {', '.join(found_skills) if found_skills else 'None'}
Missing skills: {', '.join(missing_skills) if missing_skills else 'None'}

Resume snippet (truncated):
{resume_short}

Write a concise 2-3 sentence feedback describing strengths and areas for improvement.
"""
    try:
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
        outputs = llm_model.generate(
            **inputs, max_new_tokens=120, do_sample=True, temperature=0.7, top_p=0.9
        )
        feedback = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return feedback
    except Exception as e:
        return f"Error generating feedback: {e}"

# ------------------------ STREAMLIT UI ------------------------
st.title("📄 Resume Relevance Checker")
st.markdown("Upload resumes and paste a Job Description to analyze relevance, skill gaps, and AI feedback.")

tab1, tab2 = st.tabs(["📝 Analyze Resumes", "📊 Dashboard"])

# ------------------------ TAB 1 ------------------------
with tab1:
    col1, col2 = st.columns([1,1])
    with col1:
        jd_text = st.text_area("Paste Job Description", height=250, placeholder="Paste job description here...")
    with col2:
        uploaded_files = st.file_uploader("Upload Resumes (PDF/DOCX)", type=["pdf","docx"], accept_multiple_files=True)
    
    if st.button("Try Sample JD"):
        jd_text = """Data Scientist Position
Requirements:
- Python programming and data analysis
- SQL database experience
- Machine learning and statistical modeling
- Experience with Tableau or PowerBI
- Bachelor's degree in Computer Science or related field
- 2+ years of data analysis experience"""

    if uploaded_files and jd_text:
        all_results = []
        embeddings = []
        for file in uploaded_files:
            name = file.name
            resume_text = extract_text_from_file(file)
            if resume_text:
                score = compute_relevance(resume_text, jd_text)
                found_skills = extract_skills(resume_text)
                jd_skills = extract_skills(jd_text)
                missing_skills = jd_skills - found_skills
                verdict = get_verdict(score)
                entities = extract_entities(resume_text)
                ai_feedback = generate_ai_feedback(resume_text, jd_text, found_skills, missing_skills)
                result = {
                    "Candidate": name,
                    "Score": score,
                    "Verdict": verdict,
                    "Strengths": ", ".join(sorted(found_skills)) if found_skills else "None",
                    "Missing Skills": ", ".join(sorted(missing_skills)) if missing_skills else "None",
                    "Summary": short_summary(resume_text),
                    "Emails": ", ".join(entities["Emails"]) if entities["Emails"] else "None",
                    "Phones": ", ".join(entities["Phones"]) if entities["Phones"] else "None",
                    "Education": ", ".join(entities["Education"]) if entities["Education"] else "None",
                    "Organizations": ", ".join(entities["Organizations"]) if entities["Organizations"] else "None",
                    "AI Feedback": ai_feedback
                }
                all_results.append(result)
                embeddings.append(model.encode(resume_text, convert_to_tensor=True))
        save_to_db(all_results, jd_text)

        # Display table
        st.subheader("Resume Evaluation Results")
        df_results = pd.DataFrame(all_results)
        st.dataframe(df_results)

        # Score chart
        st.subheader("Candidate Scores")
        fig, ax = plt.subplots(figsize=(8,4))
        ax.bar(df_results["Candidate"], df_results["Score"], color="royalblue")
        ax.set_ylabel("Score (%)")
        ax.set_title("Candidate Relevance Scores")
        plt.xticks(rotation=30, ha="right")
        st.pyplot(fig)

        # Candidate summaries
        st.subheader("Candidate Summaries")
        for r in all_results:
            st.markdown(f"### 💼 {r['Candidate']}")
            st.markdown(f"**Relevance Score:** {r['Score']}%  \n"
                        f"**Verdict:** {r['Verdict']}  \n"
                        f"**Strengths:** {r['Strengths']}  \n"
                        f"**Skill Gaps:** {r['Missing Skills']}  \n"
                        f"**Emails:** {r['Emails']}  \n"
                        f"**Phones:** {r['Phones']}  \n"
                        f"**Education:** {r['Education']}  \n"
                        f"**Organizations:** {r['Organizations']}  \n"
                        f"**AI Feedback:** {r['AI Feedback']}  \n"
                        f"**Summary:** {r['Summary']}")
            if r["Missing Skills"] != "None":
                st.info(f"Recommended Skills to Learn: {r['Missing Skills']}")

        # Candidate similarity
        st.subheader("Candidate Similarity Table")
        sim_matrix = util.cos_sim(torch.stack(embeddings), torch.stack(embeddings)).cpu().numpy()
        df_sim = pd.DataFrame(sim_matrix, index=df_results["Candidate"], columns=df_results["Candidate"])
        st.dataframe(df_sim.round(2))

        # Top candidates
        st.subheader("Top Recommended Candidates")
        max_candidates = len(all_results)
        top_n = st.slider("Select top N candidates", 1, max_candidates, min(3,max_candidates))
        top_sorted = sorted(all_results, key=lambda x: x["Score"], reverse=True)[:top_n]
        for r in top_sorted:
            st.write(f"• {r['Candidate']} — Score: {r['Score']}%")

        # Export CSV
        st.subheader("Download Analysis Results")
        df_export = df_results.copy()
        df_export["AI Summary"] = [
            f"Strengths: {r['Strengths']}; Improvement Areas: {r['Missing Skills']}; AI Feedback: {r['AI Feedback']}; Short Summary: {r['Summary']}" 
            for r in all_results
        ]
        csv = df_export.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", data=csv, file_name="resume_analysis.csv", mime="text/csv")
    else:
        st.info("👆 Upload at least one resume and paste a job description to analyze.")

# ------------------------ TAB 2: DASHBOARD ------------------------
with tab2:
    st.header("📊 Resume Database Dashboard")
    df_db = load_db()
    if not df_db.empty:
        st.markdown("### Overview Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Candidates", len(df_db))
        col2.metric("High Score", len(df_db[df_db["Verdict"]=="High"]))
        col3.metric("Medium Score", len(df_db[df_db["Verdict"]=="Medium"]))

        # Filter by verdict
        verdict_filter = st.multiselect("Filter by Verdict", options=df_db["Verdict"].unique())
        df_filtered = df_db[df_db["Verdict"].isin(verdict_filter)] if verdict_filter else df_db
        st.dataframe(df_filtered)

        # Candidate Scores chart
        st.markdown("### Candidate Scores Chart")
        fig, ax = plt.subplots(figsize=(10,5))
        colors = ['#4CAF50' if v=="High" else '#FFC107' if v=="Medium" else '#F44336' for v in df_filtered["Verdict"]]
        ax.bar(df_filtered["Candidate"], df_filtered["Score"], color=colors)
        ax.set_ylabel("Score (%)")
        ax.set_title("Candidate Scores by Verdict")
        plt.xticks(rotation=30)
        st.pyplot(fig)

        # Top skills
        st.markdown("### Top Skills Distribution")
        all_skills = sum([s.split(", ") for s in df_filtered["Strengths"]], [])
        skill_counts = pd.Series(all_skills).value_counts().head(10)
        st.bar_chart(skill_counts)
    else:
        st.info("Database is empty. Analyze resumes first to populate the dashboard.")
