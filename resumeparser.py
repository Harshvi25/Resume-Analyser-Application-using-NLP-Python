import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("D:\\Harshvi_MSCAIML\\AIML_Pojects\\ResumeParser\\archive\\UpdatedResumeDataSet.csv")
print(df.head())
print()

# exploring categories

print(df['Category'].value_counts())
print()

plt.figure(figsize = (10,5))
sns.countplot(df['Category'])
plt.show()
print()

print(df['Category'].unique())
category_counts = df['Category'].value_counts()
labels = df['Category'].unique()
print(df)
print()

plt.pie(category_counts,labels = labels,autopct = "%1.1f%%")
plt.show()

# exploring resume

print(df['Category'][0])
print(df['Resume'][90])

# Data Cleaning (URLs,#,Mentions,Special Letters,Punctuations)

''' for data cleaning we'll use one library which is re(Regular Expression) '''

import re

def CleanResume(txt):
    cleanTxt = re.sub("http\S+\s"," ",txt)  # for URLs
    cleanTxt = re.sub("RT|CC"," ",cleanTxt)
    cleanTxt = re.sub("#\S+"," ",cleanTxt)
    cleanTxt = re.sub("@\S+"," ",cleanTxt)    # Hashtags
    cleanTxt = re.sub("[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~""")," ",cleanTxt)
    cleanTxt = re.sub(r'[^\x00-\x7f]'," ",cleanTxt)
    cleanTxt = re.sub("\s+"," ",cleanTxt)   # it'll remove \n,\r,\t etc.

    return cleanTxt
    
print(CleanResume("my #### # $ #harshvi website like this is http://helloworld and access it @gmail.com"))
print()

df['Resume'] = df['Resume'].apply(lambda x: CleanResume(x))
print(df)

# convert words into categorical values

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Category'] = le.fit_transform(df['Category'])
print()
print(df)

# vactorization

from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer(stop_words = 'english')
RequiredText = tfid.fit_transform(df['Resume'])
# print(type(RequiredText)) matrix type

# splitting dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(RequiredText,df['Category'],test_size = 0.2,random_state = 42)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

kn = KNeighborsClassifier()
kn.fit(x_train,y_train)
pred = kn.predict(x_test)
#print(pred)
ac = accuracy_score(y_test,pred)
print(ac)

# for website

import pickle

pickle.dump(tfid,open('tfid.pkl','wb'))
pickle.dump(kn,open('kn.pkl','wb'))


# try to predict using this information in website

myResume = '''Artificial Intelligence (AI) is a field of computer science that focuses on developing systems capable of
            performing tasks that typically require human intelligence. These tasks include machine learning,
            natural language processing, computer vision, robotics, and decision-making. AI leverages algorithms, data,
            and computational power to create solutions that automate processes, enhance efficiency, and drive innovation
            across industries such as healthcare, finance, manufacturing, and more.

            Email : harshvip2002@gmail.com
            Github : https://github.com/Harshvi25?tab=repositories
            LinkdIn : www.linkedin.com/in/harshvi-patel508452255
            Phone No. : 8320250575
            Languages : English,Hindi,Gujarati

            ● Programming Language: Python, java, C++ 
            ● Machine Learning: Scikit-learn, TensorFlow, NLTK, TF-IDF 
            ● Data Analysis and Visualization: Pandas, NumPy, matplotlib, word cloud  
            ● Databases: SQL 
            ● Team Work 
            ● Communication 
            ● Quick Learner 
            ● Self-Motivated 
            ● Adaptability

            EDUCATION  
            Bachelor of Computer Application  
            Veer Narmad South Gujarat University 
            2023 
            Master of science in Artificial Intelligence & Machine Learning 
            Veer Narmad South Gujarat University 
            Pursuing 
            '''

# load the trained classifier
kn = pickle.load(open('kn.pkl','rb'))

# clean the input resume
cleaned_resume = CleanResume(myResume)

# transform the cleaned resume using the trained TfidVectorizer
input_features  = tfid.transform([cleaned_resume])

# make the prediction using the loaded classifier
prediction_id = kn.predict(input_features)[0]

# map category ID to category name

category_mapping = {
                        15 : "Java Developer",
                        23 : "Testing",
                        8  : "Devops Engineer",
                        20 : "Python Developer",
                        24 : "Web Designing",
                        12 : "HR",
                        13 : "Hadoop",
                        3  : "Blockchain",
                        10 : "ETL Developer",
                        18 : "Operations Manager",
                        6  : "Data Science",
                        22 : "Sales",
                        16 : "Mechanical Engineering",
                        1  : "Arts",
                        7  : "Database",
                        11 : "Electrical Engineering",
                        14 : "Health and fitness",
                        19 : "PMO",
                        4  : "Business Analyst",
                        9  : "DotNet Developer",
                        2  : "Automation Testing",
                        17 : "Network Security Engineering",
                        21 : "SAP Developer",
                        5  : "Civil Engineering",
                        0  : "Advocate"
                        
                   }

category_name = category_mapping.get(prediction_id,"Unknown")
print("Predicted Category : ",category_name)
print(prediction_id)







