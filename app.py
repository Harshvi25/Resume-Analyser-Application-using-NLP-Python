import streamlit as st
import pickle
import re
import nltk
from PyPDF2 import PdfReader  # For PDF files

nltk.download('punkt')
nltk.download('stopwords')

# loading models
kn = pickle.load(open('kn.pkl','rb'))
tfid = pickle.load(open('tfid.pkl','rb'))

def CleanResume(txt):
    cleanTxt = re.sub("http\S+\s"," ",txt)  # for URLs
    cleanTxt = re.sub("RT|CC"," ",cleanTxt)
    cleanTxt = re.sub("#\S+"," ",cleanTxt)
    cleanTxt = re.sub("@\S+"," ",cleanTxt)    # Hashtags
    cleanTxt = re.sub("[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")," ",cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]'," ",cleanTxt)
    cleanTxt = re.sub("\s+"," ",cleanTxt)   # it'll remove \n,\r,\t etc.

    return cleanTxt

# Extract text from uploaded files
def extract_text(upload_file):
    file_type = upload_file.name.split('.')[-1].lower()  # Get the file extension
    if file_type == 'pdf':
        try:
            pdf_reader = PdfReader(upload_file)
            return " ".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())  # Extract text
        except Exception as e:
            st.error(f"Error processing PDF file: {e}")
            return None
    elif file_type == 'txt':
        try:
            return upload_file.read().decode('utf-8')  # For text files
        except UnicodeDecodeError:
            return upload_file.read().decode('latin-1')  # Alternative decoding for text files
    else:
        st.error("Unsupported file type. Please upload a PDF or TXT file.")
        return None


# Web App
def main():
    st.title("Resume Screening App")
    upload_file = st.file_uploader("Upload Resume", type=['txt', 'pdf'])

    if upload_file is not None:
        resume_text = extract_text(upload_file)  # Extract text from uploaded file
        if resume_text:
            cleaned_resume = CleanResume(resume_text)  # Clean the extracted text
            input_features = tfid.transform([cleaned_resume])  # Transform into TF-IDF features
            prediction_id = kn.predict(input_features)[0]  # Predict category

            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "Devops Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineering",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineering",
                21: "SAP Developer",
                5: "Civil Engineering",
                0: "Advocate",
            }

            category_name = category_mapping.get(prediction_id, "Unknown")  # Get category name
            st.write(f"Predicted Category: {category_name}")  # Display the predicted category
        else:
            st.error("Could not extract text from the uploaded file. Please try again with a valid file.")

if __name__ == "__main__":
    main()
