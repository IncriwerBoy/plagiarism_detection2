import streamlit as st
from helper import Pipeline
import pickle

st.title('Plagarism Detection')

option = st.selectbox(
    "Choose your input method",
    ("Upload files", "Paste text")
)

with open('artifacts\model_category.pkl' , 'rb') as model:
    rf_category = pickle.load(model)

with open('artifacts\model_class.pkl' , 'rb') as model:
    xgb_class = pickle.load(model)


if option == "Upload files":
    col1, col2 = st.columns(2)

    with col1:
        source_file = st.file_uploader("Upload Source File", type="txt", key="source_file")
    with col2:
        answer_file = st.file_uploader("Upload Answer File", type="txt", key="answer_file")
    
    if st.button("Check Plagiarism"):
        if source_file and answer_file:
            source_text = source_file.read().decode("utf-8")
            answer_text = answer_file.read().decode("utf-8")
            
            pipeline = Pipeline(answer_text, source_text)
            
            pred1 = pipeline.plag_detection(xgb_class)
            pred2 = pipeline.plagType_detection(pred1, rf_category)
            
            if pred1 == 1:
                st.subheader("No Plagiarism Detected!")
            
            elif pred1 == 2:
                st.subheader("Plagiarism Detected!")
                val = -1
                if pred2 == 0:
                    val = "Original"
                elif pred2 == 1:
                    val = "Not Plagiarism"
                elif pred2 == 4:
                    val = 'Cut & Paste'
                elif pred2 == 2:
                    val = 'Heavy'
                elif pred2 == 3:
                    val = 'Light'
                
                st.text("Plagiarism level : " + val)
            
            elif pred1 == 0:
                st.subheader("Original Text")
        
        else:
            st.error("Please upload both source and answer files.")

elif option == "Paste text":
    col1, col2 = st.columns(2)

    with col1:
        source_text = st.text_area("Paste Source Text", height=200, key="source_text")
    with col2:
        answer_text = st.text_area("Paste Answer Text", height=200, key="answer_text")
    
    if st.button("Check Plagiarism"):
        if source_text and answer_text:
            pipeline = Pipeline(answer_text, source_text)
            
            pred1 = pipeline.plag_detection(xgb_class)
            pred2 = pipeline.plagType_detection(pred1, rf_category)
            
            if pred1 == 1:
                st.subheader("No Plagiarism Detected!")
            
            elif pred1 == 2:
                st.subheader("Plagiarism Detected!")
                val = -1
                if pred2 == 0:
                    val = "Original"
                elif pred2 == 1:
                    val = "Not Plagiarism"
                elif pred2 == 4:
                    val = 'Cut & Paste'
                elif pred2 == 2:
                    val = 'Heavy'
                elif pred2 == 3:
                    val = 'Light'
                
                st.text("Plagiarism level : " + val)
            
            elif pred1 == 0:
                st.subheader("Original Text")
        
        else:
            st.error("Please paste both source and answer texts.")