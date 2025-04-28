import fitz  # PyMuPDF
import re
import pandas as pd
import spacy
import streamlit as st
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import jsonlines
import os
import io
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi, HfFolder, login
import numpy as np
from dotenv import load_dotenv


load_dotenv()
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token)
else:
    print("Error: Hugging Face token not found.")

spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Add the EntityRuler to tag universities as COLLEGE
df = pd.read_csv("world-universities.csv")
universities = df[df.columns[0]].dropna().tolist()
college_patterns = [{"label": "COLLEGE", "pattern": uni} for uni in universities]

# Add the EntityRuler to tag skills as SKILL
skills = "jz_skill_patterns.jsonl"

with jsonlines.open(skills) as reader:
    skill_patterns = [obj for obj in reader]

if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(skill_patterns)
    ruler.add_patterns(college_patterns)

def convert_to_serializable(data):
    if isinstance(data, np.float32):
        return float(data)  # Convert np.float32 to Python float
    elif isinstance(data, dict):
        return {key: convert_to_serializable(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_serializable(item) for item in data]
    else:
        return data

common_job_titles = [

        # Computer Science
        "Software Engineer",
        "Software Developer",
        "Computer Scientist",
        "Systems Engineer",
        "Application Developer",
        "Software Architect",
        "Embedded Systems Engineer",
        "DevOps Engineer",
        "Site Reliability Engineer",
        "Solutions Architect",
        "Database Administrator",
        "Security Engineer",
        "QA Engineer",
        "Test Engineer",
        "Cloud Engineer",
        "Cloud Solutions Architect",
        "Technical Support Engineer",
        "Network Engineer",

        # Web Development
        "Frontend Developer",
        "Backend Developer",
        "Full Stack Developer",
        "Web Developer",
        "UI Developer",
        "UX Engineer",
        "JavaScript Developer",
        "React Developer",
        "Angular Developer",
        "Vue.js Developer",
        "WordPress Developer",
        "Shopify Developer",
        "PHP Developer",
        "HTML Developer",
        "CSS Developer",

        # Data Science (DS) / Data Analytics (DA)
        "Data Scientist",
        "Data Analyst",
        "Business Intelligence Analyst",
        "Machine Learning Engineer",
        "Data Engineer",
        "Big Data Engineer",
        "AI Engineer",
        "Research Scientist",
        "Statistician",
        "Business Analyst",
        "Quantitative Analyst",
        "Data Visualization Engineer",
        "Product Data Analyst",
        "Applied Scientist",

        # Machine Learning (ML) / Deep Learning (DL)
        "Machine Learning Scientist",
        "Deep Learning Engineer",
        "Computer Vision Engineer",
        "NLP Engineer",
        "Artificial Intelligence Engineer",
        "Research Engineer",
        "Reinforcement Learning Engineer",
        "Speech Recognition Engineer",
        "AI Researcher",
        "Robotics Engineer",

        # Specialized Roles
        "Mobile App Developer",
        "Android Developer",
        "iOS Developer",
        "Flutter Developer",
        "React Native Developer",
        "Blockchain Developer",
        "Cybersecurity Analyst",
        "Ethical Hacker",
        "Information Security Analyst",
        "Game Developer",
        "AR/VR Developer",

        # Management / Architect Roles
        "Technical Lead",
        "Engineering Manager",
        "Software Development Manager",
        "Project Manager",
        "Product Manager",
        "Technical Program Manager",
        "Solutions Engineer",
        "Enterprise Architect",

        # Extra Emerging Roles
        "MLOps Engineer",
        "Prompt Engineer",
        "AI Product Manager",
        "Data Quality Analyst",
        "Cloud Data Engineer",
        "Analytics Engineer",
        "DevSecOps Engineer",
        "Security Researcher",
        "Generative AI Engineer",
        "Chatbot Developer",
    ]   

@st.cache_data
def load_resume_file(uploaded_file):
    if isinstance(uploaded_file, io.BytesIO):
        return uploaded_file.getvalue()
    else:
        return uploaded_file.read()


def extract_text_from_pdf(resume_file):
    text = ""
    resume_bytes = load_resume_file(resume_file)  # ✅ Use cached loading
    doc = fitz.open(stream=resume_bytes, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_txt(jd_file):
    text = ""
    with open(jd_file, "r") as file:
        text = file.read()
    return text

def extract_name(text):
    name = re.findall(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)+', text, re.MULTILINE)
    return name[0] if name else None

def extract_email(text):
    email = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
    return email[0] if email else None

def extract_phone(text):
    phone = re.findall(r'\b\d{10}\b', text)
    return phone[0] if phone else None

def extract_skills(text):
    doc = nlp(text)
    skills = [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
    return list(set(skills))


def extract_university(text):
    field_of_study = [
        "Algorithms and Data Structures",
        "Computer Architecture and Organization",
        "Programming Languages",
        "Software Engineering",
        "Theory of Computation",
        "Artificial Intelligence (AI)",
        "Data Science",
        "Cybersecurity",
        "Cloud Computing",
        "Human-Computer Interaction (HCI)",
        "Computer Networks",
        "Database Management Systems",
        "Information Technology (IT)",
        "Game Design",
        "Animation",
        "Machine Learning",
        "Deep Learning",
        "Robotics",
        "Scientific Computing"
    ]
    values_college = []
    values_field = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ == 'COLLEGE' or ent.label_ == "ORG" and (
                "University" in ent.text or "Institute" in ent.text or "College" in ent.text
        ):
            values_college.append(ent.text)

        if ent.text in field_of_study:
            values_field.append(ent.text)
    manual_matches = re.findall(r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s(?:University|Institute|College))\b', text)
    values_college.extend(manual_matches)
    return values_college[0] if values_college else None, values_field[0] if values_field else None


def extract_certifications(text):
    certifications = []
    for line in text.split('\n'):
        line = line.strip()
        cert_keywords = ["certified", "certificate", "certification", "license", "AWS", "Azure"]
        if any(keyword.lower() in line.lower() for keyword in cert_keywords):
            certifications.append(line.lower())

    return certifications if certifications else None

def structure(resume_text):
    name = extract_name(resume_text)
    email = extract_email(resume_text)
    phone = extract_phone(resume_text)
    skills = extract_skills(resume_text)
    certifications = extract_certifications(resume_text)
    university = extract_university(resume_text)

    return name, email, phone, skills, certifications, university

def save_to_huggingface_dataset(parsed_data):
    dataset_path = "resumes_data.jsonl"
    repo_id = "mayankpuvvala/resume_parser_genai"  # Replace with your Hugging Face username and repo name

    # Convert any non-serializable data (like np.float32) to serializable data types
    parsed_data = convert_to_serializable(parsed_data)

    # Save parsed data to a local JSONL file
    with open(dataset_path, "a") as f:
        json.dump(parsed_data, f)
        f.write("\n")

    # Load the JSONL file into a Hugging Face Dataset
    dataset = Dataset.from_json(dataset_path)

    # Push the dataset to the Hugging Face Hub
    dataset.push_to_hub(repo_id, private=True)

    st.success("✅ Resume saved successfully to HuggingFace Dataset format.")


def read_resume(resume_file, jd_text):
    resume_text = extract_text_from_pdf(resume_file)
    if len(resume_text) == 0 or len(jd_text) == 0:
        st.error("Please upload a valid resume and job description.")
        return None

    inputs_resume = tokenizer.batch_encode_plus([resume_text], return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs_jd = tokenizer.batch_encode_plus([jd_text], return_tensors="pt", max_length=512, truncation=True, padding="max_length")

    outputs_resume = model(**inputs_resume)
    outputs_jd = model(**inputs_jd)

    cls_resume = outputs_resume.last_hidden_state[:, 0, :]
    cls_jd = outputs_jd.last_hidden_state[:, 0, :]
    # compare both outputs using bert, and get the cosine similarity score
    similarity_score = cosine_similarity(cls_resume.detach().numpy(), cls_jd.detach().numpy())[0][0]

    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    matched_skills = (set(jd_skills) & set(resume_skills))
    missing_skills = (set(jd_skills) - set(resume_skills))

    st.write("✅ Matched Skills:", matched_skills)
    st.write("❌ Missing Skills:", missing_skills)

    name, email, phone, skills, certifications, university = structure(resume_text)

    parsed_data = {
        "Name": name,
        "Email": email,
        "Phone": phone,
        "Skills": ",".join(skills) if skills else None,
        "Certifications": ",".join(certifications) if certifications else None,
        "University": university[0] if university and university[0] else None,
        "Similarity Score": similarity_score
    }

    save_to_huggingface_dataset(parsed_data)

    return similarity_score


st.title("Resume Parser and Scoring")

resume_file = st.file_uploader("Upload Resume", type=["pdf"])
jd_text = st.text_area("Paste the Job Description", height=200)

if st.button("Process Resume", disabled=not (resume_file and jd_text)):
    score = read_resume(resume_file, jd_text)
    if score is not None:
        st.success(f"Resume Match Score: {round(score * 100, 2)}%") 
        if st.button("Load Dataset"):
            try:
                dataset = load_dataset("mayankpuvvala/resume_parser_genai")
                st.dataframe(dataset['train'].to_pandas())
                st.write(dataset)
            except Exception as e:
                st.error(f"Failed to load dataset: {e}")
