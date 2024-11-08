import tkinter as tk
from tkinter import messagebox, ttk, filedialog
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
import re
import PyPDF2

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Load the job dataset from the CSV file
job_data = pd.read_csv('job_postings_1000.csv')

# Function to close the application
def close_app():
    if messagebox.askokcancel("Quit", "Do you want to quit?"):
        root.destroy()

# Enhanced resume parsing function to handle more complex resume structures
def parse_resume():
    file_path = filedialog.askopenfilename(filetypes=[("PDF files", "*.pdf")])
    if file_path:
        # Extract text from the PDF file
        resume_text = extract_text_from_pdf(file_path)

        # Extract education
        extract_education(resume_text)

        # Extract skills
        extract_skills(resume_text)

        # Extract professional experience
        extract_experience(resume_text)

        # Extract projects
        extract_projects(resume_text)

        # Extract certifications
        extract_certifications(resume_text)

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(file_path):
    text = ''
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in range(len(reader.pages)):
            text += reader.pages[page].extract_text()
    return text

# Extract education details
def extract_education(resume_text):
    education_pattern = r'EDUCATION\s*(.+?)\s*(?=SKILLS|PROFESSIONAL EXPERIENCE|PROJECTS|CERTIFICATES)'
    education_match = re.search(education_pattern, resume_text, re.DOTALL | re.IGNORECASE)
    if education_match:
        education_text = education_match.group(1).strip()
        education_entry.delete(0, tk.END)
        education_entry.insert(0, education_text)

# Extract skills using a robust regex pattern
def extract_skills(resume_text):
    skills_pattern = r'SKILLS\s*(.+?)\s*(?=PROFESSIONAL EXPERIENCE|PROJECTS|CERTIFICATES)'
    skills_match = re.search(skills_pattern, resume_text, re.DOTALL | re.IGNORECASE)
    if skills_match:
        skills_text = skills_match.group(1).strip()
        skills_entry.delete(0, tk.END)
        skills_entry.insert(0, skills_text)

# Extract professional experience
def extract_experience(resume_text):
    experience_pattern = r'PROFESSIONAL EXPERIENCE\s*(.+?)\s*(?=PROJECTS|CERTIFICATES|ACHIEVEMENTS)'
    experience_match = re.search(experience_pattern, resume_text, re.DOTALL | re.IGNORECASE)
    if experience_match:
        experience_text = experience_match.group(1).strip()
        experience_entry.delete(0, tk.END)
        experience_entry.insert(0, experience_text)

# Extract project names and brief descriptions
def extract_projects(resume_text):
    projects_pattern = r'PROJECTS\s*(.+?)\s*(?=CERTIFICATES|ACHIEVEMENTS|$)'
    projects_match = re.search(projects_pattern, resume_text, re.DOTALL | re.IGNORECASE)
    if projects_match:
        projects_text = projects_match.group(1).strip()
        projects_entry.delete(0, tk.END)
        projects_entry.insert(0, projects_text)

# Extract certification names
def extract_certifications(resume_text):
    certifications_pattern = r'CERTIFICATES\s*(.+?)\s*(?=ACHIEVEMENTS|$)'
    certifications_match = re.search(certifications_pattern, resume_text, re.DOTALL | re.IGNORECASE)
    if certifications_match:
        certifications_text = certifications_match.group(1).strip()
        certifications_entry.delete(0, tk.END)
        certifications_entry.insert(0, certifications_text)

# Function to recommend jobs
def recommend_jobs():
    user_skills = skills_entry.get()
    user_experience_years = experience_years_combo.get()
    user_experience_field = experience_field_combo.get()
    user_location = location_combo.get()
    salary_min = int(salary_min_entry.get()) if salary_min_entry.get().isdigit() else 0
    salary_max = int(salary_max_entry.get()) if salary_max_entry.get().isdigit() else float('inf')
    selected_industry = industry_combo.get()
    selected_job_type = job_type_combo.get()

    if not user_skills or not user_experience_years or not user_experience_field:
        messagebox.showwarning("Input Error", "Please fill in all the fields")
        return

    user_profile = f"{user_skills} {user_experience_years} years of experience in {user_experience_field}"

    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'), max_features=100)
    job_vectors = vectorizer.fit_transform(job_data['skills'] + " " + job_data['description'])
    user_vector = vectorizer.transform([user_profile])

    # Cosine Similarity
    similarity_matrix = cosine_similarity(user_vector, job_vectors)
    similar_jobs = similarity_matrix.argsort().flatten()[::-1]

    # Filter by location, salary, industry, and job type
    filtered_jobs = []
    for idx in similar_jobs:
        job = job_data.iloc[idx]
        if (job['location'] == user_location or not user_location) and \
           (salary_min <= job['salary'] <= salary_max) and \
           (job['industry'] == selected_industry or selected_industry == 'Any') and \
           (job['job_type'] == selected_job_type or selected_job_type == 'Any'):
            filtered_jobs.append(idx)
            if len(filtered_jobs) >= 5:  # Limit to top 5 recommendations
                break

    # Display recommended jobs
    recommendations_text.delete(1.0, tk.END)  # Clear previous recommendations
    for idx in filtered_jobs:
        title = job_data.iloc[idx]['title']
        description = job_data.iloc[idx]['description']
        skills = job_data.iloc[idx]['skills']
        location = job_data.iloc[idx]['location']
        salary = job_data.iloc[idx]['salary']
        industry = job_data.iloc[idx]['industry']
        job_type = job_data.iloc[idx]['job_type']
        recommendations_text.insert(tk.END, f"Title: {title}\nDescription: {description}\nSkills: {skills}\n"
                                            f"Location: {location}\nSalary: ${salary}\nIndustry: {industry}\n"
                                            f"Job Type: {job_type}\n\n")

# Creating the main window
root = tk.Tk()
root.title("Job Recommendation System")

# Making the window full screen
root.attributes('-fullscreen', True)

# Creating the close button in the top-right corner
close_button = tk.Button(root, text="Close Application", command=close_app, bg='red', fg='white')
close_button.grid(row=0, column=2, sticky=tk.NE, padx=10, pady=10)

# Creating labels and entry fields for skills and experience, placed in the upper left side
skills_label = tk.Label(root, text="Enter your skills (comma-separated):")
skills_label.grid(row=0, column=0, sticky=tk.W, padx=10, pady=5)

skills_entry = tk.Entry(root, width=50)
skills_entry.grid(row=0, column=1, padx=10, pady=5)

# Dropdown for Years of Experience
experience_label = tk.Label(root, text="Years of Experience:")
experience_label.grid(row=1, column=0, sticky=tk.W, padx=10, pady=5)

experience_years_combo = ttk.Combobox(root, values=[str(i) for i in range(1, 21)])
experience_years_combo.grid(row=1, column=1, padx=10, pady=5, sticky=tk.W)

# Dropdown for Field of Experience
experience_field_label = tk.Label(root, text="Field of Experience:")
experience_field_label.grid(row=1, column=2, sticky=tk.W, padx=10, pady=5)

experience_field_combo = ttk.Combobox(root, values=list(job_data['industry'].unique()))
experience_field_combo.grid(row=1, column=3, padx=10, pady=5, sticky=tk.W)

# Dropdown for Preferred Location
location_label = tk.Label(root, text="Preferred Location:")
location_label.grid(row=2, column=0, sticky=tk.W, padx=10, pady=5)

location_combo = ttk.Combobox(root, values=list(job_data['location'].unique()))
location_combo.grid(row=2, column=1, padx=10, pady=5, sticky=tk.W)

# Adding salary range filter
salary_label = tk.Label(root, text="Preferred Salary Range:")
salary_label.grid(row=3, column=0, sticky=tk.W, padx=10, pady=5)

salary_min_label = tk.Label(root, text="Min Salary:")
salary_min_label.grid(row=4, column=0, sticky=tk.W, padx=10, pady=5)
salary_min_entry = tk.Entry(root, width=20)
salary_min_entry.grid(row=4, column=1, padx=10, pady=5, sticky=tk.W)

salary_max_label = tk.Label(root, text="Max Salary:")
salary_max_label.grid(row=5, column=0, sticky=tk.W, padx=10, pady=5)
salary_max_entry = tk.Entry(root, width=20)
salary_max_entry.grid(row=5, column=1, padx=10, pady=5, sticky=tk.W)

# Adding industry filter
industry_label = tk.Label(root, text="Preferred Industry:")
industry_label.grid(row=6, column=0, sticky=tk.W, padx=10, pady=5)

industry_combo = ttk.Combobox(root, values=['Any', 'Finance', 'Healthcare', 'Education', 'Engineering', 'Technology'])
industry_combo.set('Any')
industry_combo.grid(row=6, column=1, padx=10, pady=5, sticky=tk.W)

# Adding job type filter
job_type_label = tk.Label(root, text="Preferred Job Type:")
job_type_label.grid(row=7, column=0, sticky=tk.W, padx=10, pady=5)

job_type_combo = ttk.Combobox(root, values=['Any', 'Full-Time', 'Part-Time', 'Contract', 'Remote'])
job_type_combo.set('Any')
job_type_combo.grid(row=7, column=1, padx=10, pady=5, sticky=tk.W)

# Adding fields for extracted projects and certifications
projects_label = tk.Label(root, text="Extracted Projects:")
projects_label.grid(row=8, column=0, sticky=tk.W, padx=10, pady=5)

projects_entry = tk.Entry(root, width=50)
projects_entry.grid(row=8, column=1, padx=10, pady=5)

certifications_label = tk.Label(root, text="Extracted Certifications:")
certifications_label.grid(row=9, column=0, sticky=tk.W, padx=10, pady=5)

certifications_entry = tk.Entry(root, width=50)
certifications_entry.grid(row=9, column=1, padx=10, pady=5)

# Create the button to upload resume and parse it
upload_button = tk.Button(root, text="Upload Resume", command=parse_resume, bg='blue', fg='white')
upload_button.grid(row=10, column=0, padx=10, pady=10, sticky=tk.W)

# Create the button to get job recommendations
recommend_button = tk.Button(root, text="Get Recommendations", command=recommend_jobs, bg='green', fg='white')
recommend_button.grid(row=10, column=1, padx=10, pady=10, sticky=tk.W)

# Text box to display job recommendations
recommendations_text = tk.Text(root, height=10, width=100)
recommendations_text.grid(row=11, column=0, columnspan=4, padx=10, pady=10)

# Start the main loop of the Tkinter application
root.mainloop()
