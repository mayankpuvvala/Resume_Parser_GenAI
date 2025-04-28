import fitz # PyMuPDF
import re
import pandas as pd
import spacy
import mysql.connector
from mysql.connector import Error
import streamlit as st
from collections import defaultdict
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch 
from difflib import SequenceMatcher
import json
import jsonlines
import matplotlib.pyplot as plt
import xlsxwriter
import json
from dotenv import load_dotenv
import os
import io

load_dotenv()  # Loads environment variables from .env file

username = os.getenv("MYSQL_USERNAME")
password = os.getenv("MYSQL_PASSWORD")

spacy.cli.download("en_core_web_sm")
nlp= spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model= BertModel.from_pretrained('bert-base-uncased')
model.eval()

# Add the EntityRuler to tag universities as COLLEGE
df = pd.read_csv("world-universities.csv")
universities = df[df.columns[0]].dropna().tolist()
college_patterns = [{"label": "COLLEGE", "pattern": uni} for uni in universities]

# Add the EntityRuler to tag skills as SKILL
skills= "jz_skill_patterns.jsonl"

with jsonlines.open(skills) as reader:
    skill_patterns = [obj for obj in reader]

if "entity_ruler" not in nlp.pipe_names:
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.add_patterns(skill_patterns)
    ruler.add_patterns(college_patterns)

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

try:
    mydb= mysql.connector.connect(
        user=username,
        password=password,
        host="localhost",
        database="mydatabase"
    )
    if mydb.is_connected():
        print("Connected to MySQL database")
        mycursor= mydb.cursor()
    else:
        print("Not connected to MySQL database")
except Error as e:
    print("Error while connecting to MySQL", e)


def extract_text_from_pdf(resume_file):
    text = ""
    if isinstance(resume_file, io.BytesIO):  # If it's a file-like object
        doc = fitz.open(stream=resume_file.read(), filetype="pdf")
    else:  # If it's a path string
        doc = fitz.open(resume_file, filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def extract_text_from_txt(jd_file):
    text= ""
    with open(jd_file, "r") as file:
        text= file.read()
    return text
def extract_name(text):
        name= re.findall(r'^[A-Z][a-z]+(?:\s[A-Z][a-z]+)+', text, re.MULTILINE)
        return name[0] if name else None
         
def extract_email(text):
        email = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text)
        return email[0] if email else None
    
def extract_phone(text):
        phone = re.findall(r'\b\d{10}\b', text)
        return phone[0] if phone else None

def extract_skills(text):
    doc= nlp(text)
    skills= [ent.text for ent in doc.ents if ent.label_ == 'SKILL']
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
        values_college= []
        values_field= []
        doc= nlp(text)
        for ent in doc.ents:
            if ent.label_== 'COLLEGE' or ent.label_ == "ORG" and (
            "University" in ent.text or "Institute" in ent.text or "College" in ent.text
        ):
                values_college.append(ent.text)
            
            if ent.text in field_of_study:
                values_field.append(ent.text)
        manual_matches = re.findall(r'\b(?:[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\s(?:University|Institute|College))\b', text)
        values_college.extend(manual_matches)
        return values_college[0] if values_college else None, values_field[0] if values_field else None
    


def extract_certifications(text):
    certifications= []
    for line in text.split('\n'):
        line= line.strip()
        cert_keywords = ["certified", "certificate", "certification", "license", "AWS", "Azure"]
        if any(keyword.lower() in line.lower() for keyword in cert_keywords):
            certifications.append(line.lower())

    return certifications if certifications else None

def structure(resume_text):
        
        name= extract_name(resume_text)
        email= extract_email(resume_text)
        phone= extract_phone(resume_text)
        skills = extract_skills(resume_text)
        certifications = extract_certifications(resume_text)
        university = extract_university(resume_text)

        return name, email, phone, skills, certifications, university

def read_resume(resume_file, jd_text):
    resume_text= extract_text_from_pdf(resume_file)
    jd_text= jd_text
    if len(resume_text)==0 or len(jd_text)==0:
        st.error("Please upload a valid resume and job description.")
        return None


    inputs_resume = tokenizer.batch_encode_plus([resume_text], return_tensors="pt",max_length=512, truncation=True, padding=True)
    inputs_jd = tokenizer.batch_encode_plus([jd_text], return_tensors= "pt",max_length=512,  truncation=True, padding=True)
    
    
    outputs_resume= model(**inputs_resume)
    outputs_jd= model(**inputs_jd)

    cls_resume = outputs_resume.last_hidden_state[:, 0, :]
    cls_jd = outputs_jd.last_hidden_state[:, 0, :]
    # compare both outputs using bert,  and get the cosine similarity score
    similarity_score = cosine_similarity(cls_resume.detach().numpy(), cls_jd.detach().numpy())[0][0]


    jd_skills = extract_skills(jd_text)
    resume_skills = extract_skills(resume_text)

    matched_skills = (set(jd_skills) & set(resume_skills))
    missing_skills = (set(jd_skills) - set(resume_skills))

    st.write("✅ Matched Skills:", matched_skills)
    st.write("❌ Missing Skills:", missing_skills)

    
    name, email, phone, skills, certifications, university = structure(resume_text)
        
    mycursor.execute('''
            CREATE TABLE IF NOT EXISTS resume (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                email VARCHAR(255),
                phone VARCHAR(20),
                skills TEXT,
                certifications TEXT,
                university TEXT
            )
''')

    try:
            sql= "INSERT INTO resume (name, email, phone, skills, certifications, university) VALUES (%s, %s, %s, %s, %s, %s)"        
            val= (name, email, phone, 
                ",".join(skills) if skills else None, 
            ",".join(certifications) if certifications else None, 
            university[0] if university and university[0] else None
)
            mycursor.execute(sql, val)
            mydb.commit()
    except Error as e:
            print("Error while inserting into MySQL", e)

    df = pd.DataFrame([{
            "Name": name,
            "Email": email,
            "Phone": phone,
            "Skills": ",".join(skills) if skills else None,
            "Certifications": ",".join(certifications) if certifications else None,
            "University": ",".join(university[0]) if university and university[0] else None,
            "Similarity Score": similarity_score
        }])

    import io

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download CSV",
        data=csv_buffer.getvalue(),
        file_name="resume_details.csv",
        mime="text/csv"
    )
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False)
    st.download_button(
        label="Download Excel",
        data=excel_buffer.getvalue(),
        file_name="resume_details.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
        
    return similarity_score


st.title("Resume Parser and Scoring")

resume_file = st.file_uploader("Upload Resume", type=["pdf"])
jd_text = st.text_area("Paste the Job Description", height=200)

if resume_file and jd_text:
    # process here
    resume_text = extract_text_from_pdf(resume_file)

if st.button("Process Resume", disabled= not (resume_file and jd_text)):
    score = read_resume(resume_file, jd_text)
    if score is not None:
        st.success(f"Resume Match Score: {round(score * 100, 2)}%")
