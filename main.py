import os
import streamlit as st
import pytesseract
from pdf2image import convert_from_bytes
from arabicTransformers import ArabicTransformers
import re

def perform_ocr(image):
    # Perform OCR on the image to extract text
    extracted_text = pytesseract.image_to_string(image, lang='ara')
    return extracted_text

def preprocess_text(text):
    # Remove non-letter characters except numbers
    text = re.sub(r'[^؀-ۿ\d]+', ' ', text)
    return text

def main():
    st.title("Arabic PDF OCR and Question Answering")

    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.read()

        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes)

        extracted_text = ""
        for page in images:
            extracted_text += perform_ocr(page)

        preprocessed_text = preprocess_text(extracted_text)
        st.write("Extracted and Preprocessed Text (Arabic):")
        st.write(preprocessed_text)

        question = st.text_input("Enter a question in Arabic:")
        if st.button("Answer Question"):
            model = ArabicTransformers('ZeyadAhmed/AraElectra-Arabic-SQuADv2-QA')
            result = ArabicTransformers.question_answering(model, question=question, context=preprocessed_text)
            
            # Accessing the appropriate key in the dictionary and splitting its value
            answer_text = result['answer']
            
            # Extracting context around the answer
            context_start = max(0, result['start'] - 100)  # Adjust window size as needed
            context_end = min(len(preprocessed_text), result['end'] + 100)  # Adjust window size as needed
            context = preprocessed_text[context_start:context_end]
            
            st.write("Answer:")
            st.write(answer_text)
            st.write("Context around the answer:")
            st.write(context)

if __name__ == "__main__":
    main()
