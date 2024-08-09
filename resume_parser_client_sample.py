import requests

def upload_file(url, file_path):
    # Open the file in binary mode
    with open(file_path, 'rb') as file:
        # Define the files in the format expected by `requests`
        files = {'file': (file_path, file, 'application/pdf')}  # Adjust the MIME type if necessary
        data = {
            'average_time_at_jobs_threshold': '24',  # example threshold
            'average_time_under_3_years_threshold': '18',
            'total_jobs_under_3_years_threshold': '5',
            'criteria_method': 2
        }
        # Make a POST request to upload the file
        response = requests.post(url, files=files, data=data)
        
        return response
    

def upload_job_fit_data(url, file_path):
    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file, 'application/pdf')}
        data = {
            'job_description': 'Data Scientist with 5 years of experience',
            'tech_keywords': 'Python, Machine Learning',
            'exp_keywords': 'Data Mining, Model Development',
            'edu_keywords': 'Bachelor in Computer Science',
            'tech_weight': 30,
            'exp_weight': 40,
            'edu_weight': 30
        }
        response = requests.post(url, files=files, data=data)
        return response

def upload_resume_and_extract_data(url, file_path, data_points):
    with open(file_path, 'rb') as file:
        # Prepare the multipart/form-data payload
        files = {
            'file': (file_path, file, 'application/pdf'),  # Adjust the MIME type if necessary
        }
        # Data points are passed as form data
        data = {'data_points': data_points}
        response = requests.post(url, files=files, data=data)
        return response

# URL of the FastAPI endpoint
url = 'http://127.0.0.1:8000/job-hopper-identifier/'

# Path to the file you want to upload
file_path = r'sample_resume.pdf'  # Change this to the path of your PDF or DOC

# Call the upload function
response = upload_file(url, file_path)

print("Status Code:", response.status_code)
print("Response Body:", response.text)

# URL of the FastAPI endpoint
url = 'http://127.0.0.1:8000/job-fit-identifier/'

# Path to the file you want to upload
file_path = r'sample_resume.pdf'  # Change this to the path of your PDF or DOC

# Call the function to upload job fit data
response = upload_job_fit_data(url, file_path)

print("Status Code:", response.status_code)
print("Response Body:", response.text)

# URL of the FastAPI endpoint
url = 'http://127.0.0.1:8000/resume-data-extraction/'

# Path to the file you want to upload
file_path = r'sample_resume.pdf'  # Change this to the actual path of your resume file

# List of data points you want to extract from the resume
data_points = ['Name', 'Email', 'Phone Number', 'Education']

# Call the upload function
response = upload_resume_and_extract_data(url, file_path, data_points)

print("Status Code:", response.status_code)
print("Response Body:", response.json())  # Ensure your API returns JSON for better formatting