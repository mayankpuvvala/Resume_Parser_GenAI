#import libraries
#upload the resume
#understand the text from resume
#enter the values in sql
#score the resume based on the job description use transformers
#return the excel sheet and the score of the resume
'''username:root
password:mysql'''
'''we are first going to upload resume and jd, and then extract text and see cosine similarity and give a score, 
and we'll say which words to add to resume, we'll use space for NER, and BERT for text understanding, 
we are going to deploy this using streamlit and database is mysql, '''

import streamlit as st
import fitz # PyMuPDF
import re
import pandas as pd
import spacy
import mysql.connector
from mysql.connector import Error
import streamlit as st

nlp= spacy.load("en_core_web_sm")
try:
    mydb= mysql.connector.connect(
        user="root",
        password="mysql",
        host="localhost",
        database="mydatabase"
    )
    if mydb.is_connected():
        print("Connected to MySQL database")
    else:
        print("Not connected to MySQL database")
except Error as e:
    print("Error while connecting to MySQL", e)

mycursor= mydb.cursor()

#upload the resume
def read_resume(file_path_resume, file_path_jd):
    text_resume= ""
    text_job= ""
    resume= fitz.open(file_path_resume)
    jd= fitz.open(file_path_jd)
    for page in resume:
        text_resume+=page.get_text()
    # print(text)
    for page in jd:
        text_job+=page.get_text()

    email = re.findall(r'[\w.+-]+@[\w-]+\.[\w.-]+', text_resume)
    print(email)

    def person(text):
        doc= nlp(text)
        for ent in doc.ents:
            if ent.label_== 'PERSON':
                print( ent.text)
                break
    name= person(text_resume)
    print(name)
    #skills, experience, keywords, and certifications

    for i in text_resume.split():
        if i=='Skills':
            print(i)
        
    mycursor.execute("SHOW TABLES")
    tables = mycursor.fetchall()

    print("Tables in the database:")
    for table in tables:
        print(table[0])


read_resume('Mayank_Puvvala(Resume).pdf','jd.txt')
