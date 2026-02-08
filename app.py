import re
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Resume–Job Match Scorer", page_icon="🧩", layout="wide")

st.title("🧩 Resume–Job Match Scorer")
st.write("Paste a resume and a job description to get a match score and see the top overlapping keywords.")

col1, col2 = st.columns(2)
with col1:
    resume_text = st.text_area("Resume Text", height=260, placeholder="Paste resume here...")
with col2:
    job_text = st.text_area("Job Description", height=260, placeholder="Paste job description here...")

clean = lambda t: re.sub(r"[^a-zA-Z0-9\s]", " ", t.lower())

if st.button("Score Match"):
    if not resume_text.strip() or not job_text.strip():
        st.warning("Please paste both resume and job description.")
    else:
        docs = [clean(resume_text), clean(job_text)]
        vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
        tfidf = vectorizer.fit_transform(docs)
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]

        st.metric("Match Score", f"{score*100:.1f}%")

        # Top overlapping terms
        feature_names = vectorizer.get_feature_names_out()
        resume_vec = tfidf[0].toarray().flatten()
        job_vec = tfidf[1].toarray().flatten()
        overlap = (resume_vec * job_vec)
        top_idx = overlap.argsort()[::-1][:15]
        top_terms = [(feature_names[i], overlap[i]) for i in top_idx if overlap[i] > 0]

        if top_terms:
            st.subheader("Top Matching Keywords")
            df = pd.DataFrame(top_terms, columns=["Keyword", "Overlap Score"])
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No strong keyword overlap found. Try adding more skills or keywords.")

st.caption("TF‑IDF based similarity scoring. For a deeper match, upgrade to embeddings later.")
