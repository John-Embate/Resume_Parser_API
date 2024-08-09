from google.generativeai.types import HarmCategory, HarmBlockThreshold
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from pdfminer.high_level import extract_text
from fastapi.openapi.docs import get_swagger_ui_html
import google.generativeai as genai
from typing import List
from dotenv import load_dotenv
from typing import Optional
import datetime
import json
import csv
import tempfile
import shutil
import os
import copy

app = FastAPI()

#General Functions: START Here

def configure_genai():
    load_dotenv(dotenv_path='cred.env')
    if os.getenv("GOOGLE_GEN_AI_API_KEY"):
        genai.configure(api_key=os.getenv("GOOGLE_GEN_AI_API_KEY"))
    else:
        raise ValueError("GOOGLE_GEN_AI_API_KEY is not set in the environment variables.")

def convert_numeric_values(data):
    if isinstance(data, dict):
        return {k: convert_numeric_values(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_numeric_values(item) for item in data]
    else:
        return convert_to_numeric(data)

def convert_to_numeric(value):
    # Attempt to convert strings that represent integers or floats to their numeric forms
    try:
        return int(value)
    except (ValueError, TypeError):
        pass
    try:
        return float(value)
    except (ValueError, TypeError):
        pass
    return value

def extract_and_parse_json(text):
    # Find the first opening and the last closing curly brackets
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None, False  # Proper JSON structure not found

    # Extract the substring that contains the JSON
    json_str = text[start_index:end_index + 1]

    try:
        # Attempt to parse the JSON
        parsed_json = json.loads(json_str)
        # Convert all numeric values if they are in string format
        parsed_json = convert_numeric_values(parsed_json)
        return parsed_json, True
    except json.JSONDecodeError:
        return None, False  # JSON parsing failed

def is_expected_json_content(json_data, type = "job_hopper"):
    """
    Validates if the passed argument is a valid JSON with the expected structure.

    Args:
        json_data: The JSON data to validate.

    Returns:
        True if the JSON is valid and has the expected structure, False otherwise.
    """
    
    try:
        # Try to load the JSON data
        data = json.loads(json_data) if isinstance(json_data, str) else json_data
    except json.JSONDecodeError:
        return False

    if type == "job_hopper":
        # Define required top-level keys for 'job_hopper'
        required_keys = ["Name", "Year Graduated", "Company History"]
        required_company_keys = ["Company Name", "Year Started", "Job Role", "Working Months"]

        if not all(key in data for key in required_keys):
            return False
        if not isinstance(data["Company History"], list):
            return False
        for company_history in data["Company History"]:
            if not all(key in company_history for key in required_company_keys):
                return False
    
    elif type == "job_fit":
        # Define required top-level keys for 'job_fit'
        required_keys = ["Name", "Technical Skills", "Relevant Experience", "Education Relevance", "Overall Score", "Overall Assessment"]
        if not all(key in data for key in required_keys):
            return False
        if not isinstance(data.get("Technical Skills"), int) or not isinstance(data.get("Relevant Experience"), int) or not isinstance(data.get("Education Relevance"), int) or not isinstance(data.get("Overall Score"), int):
            return False
        if not isinstance(data.get("Overall Assessment"), str):
            return False

    else:
        return False  # Unsupported type

    return True  # All checks passed for the specified type

def extract_and_parse_json(text):
    # Find the first opening and the last closing curly brackets
    start_index = text.find('{')
    end_index = text.rfind('}')
    
    if start_index == -1 or end_index == -1 or end_index < start_index:
        return None, False  # Proper JSON structure not found

    # Extract the substring that contains the JSON
    json_str = text[start_index:end_index + 1]

    try:
        # Attempt to parse the JSON
        parsed_json = json.loads(json_str)
        # Convert all numeric values if they are in string format
        parsed_json = convert_numeric_values(parsed_json)
        return parsed_json, True
    except json.JSONDecodeError:
        return None, False  # JSON parsing failed
    
#For Job Hopper Resume Analysis: START
def get_longest_time_at_a_job_duration_company(json_data):
    """Returns a tuple (company_name, job_role, duration) of the longest time an applicant 
    has worked based on the data of "Company History", handling potential null durations."""
    max_duration = 0
    company_name = ""
    job_role = ""
    for job in json_data["Company History"]:
        duration = job.get("Working Months")  # Use .get() to handle missing keys
        if duration is not None and duration > max_duration:
            max_duration = duration
            company_name = job["Company Name"]
            job_role = job["Job Role"]
    return company_name, job_role, max_duration if max_duration > 0 else None  # Return None if no valid duration found


def get_shortest_time_at_a_job_duration_company(json_data):
    """Returns a tuple (company_name, job_role, duration) of the shortest time an applicant 
    has worked based on the data of "Company History", handling potential null durations."""
    min_duration = float('inf')
    company_name = ""
    job_role = ""
    for job in json_data["Company History"]:
        duration = job.get("Working Months")
        if duration is not None and duration < min_duration:
            min_duration = duration
            company_name = job["Company Name"]
            job_role = job["Job Role"]
    return company_name, job_role, min_duration if min_duration != float('inf') else None 

def get_average_time_at_jobs(json_data):
    """Gets the overall average work duration per job by the applicant 
    based on company history, handling potential null durations."""
    total_months = 0
    valid_job_count = 0
    for job in json_data["Company History"]:
        duration = job.get("Working Months")
        if duration is not None:
            total_months += duration
            valid_job_count += 1
    return total_months / valid_job_count if valid_job_count > 0 else 0

def get_total_number_of_jobs(json_data):
    """Gets the total number of jobs an applicant had based on company history. 
    This function assumes all entries in "Company History" represent valid jobs."""
    return len(json_data["Company History"])

def has_worked_more_than_3_years(json_data):
    """
    Checks if the applicant has worked on any job for more than 3 years (36 months),
    handling potential null durations.
    """
    jobs_more_than_3_years = [job for job in json_data["Company History"] if job.get("Working Months") is not None and job.get("Working Months") > 36]
    has_worked_more_than_3_years = len(jobs_more_than_3_years) > 0
    if has_worked_more_than_3_years:
        average_job_stay = sum(job.get("Working Months") for job in jobs_more_than_3_years) / len(jobs_more_than_3_years)
        num_jobs = len(jobs_more_than_3_years)
        return average_job_stay, num_jobs, has_worked_more_than_3_years
    else:
        return None, None, False

def has_worked_less_than_3_years(json_data):
    """
    Checks if the applicant has worked on any job for less than 3 years (36 months),
    handling potential null durations.
    """
    jobs_less_than_3_years = [job for job in json_data["Company History"] if job.get("Working Months") is not None and job.get("Working Months") < 36]
    has_worked_less_than_3_years = len(jobs_less_than_3_years) > 0
    if has_worked_less_than_3_years:
        average_job_stay = sum(job.get("Working Months") for job in jobs_less_than_3_years) / len(jobs_less_than_3_years)
        num_jobs = len(jobs_less_than_3_years)
        return average_job_stay, num_jobs, has_worked_less_than_3_years
    else:
        return None, None, False
    
def analyze_resume(json_data):
    """Analyzes the resume data and populates an analysis dictionary.

    Args:
        json_data: The JSON data containing the resume information.

    Returns:
        A dictionary containing the analysis results.
    """

    analysis_dict = {
        "Name": json_data["Name"],
        "Total Number of Jobs": get_total_number_of_jobs(json_data),
        "Longest Time at a Job (Duration)": None,  # Placeholder
        "Longest Time at a Job (Company)": None,  # Placeholder
        "Longest Time at a Job (Job Role)": None, #Placeholder
        "Shortest Time at a Job (Duration)": None,  # Placeholder
        "Shortest Time at a Job (Company)": None,  # Placeholder
        "Shortest Time at a Job (Job Role)": None, #Placeholder
        "Average Time at Jobs": get_average_time_at_jobs(json_data),
        "Has worked for more than 3 years": None,  # Placeholder, will be updated
        "Total Number of Jobs worked for more than 3 years": None,  # Placeholder
        "Average Duration of Jobs worked for more than 3 years": None,  # Placeholder
        "Has worked for less than 3 years": None,  # Placeholder
        "Total Number of Jobs worked for less than 3 years": None,  # Placeholder
        "Average Duration of Jobs worked for less than 3 years": None  # Placeholder
    }

    # Get longest job details
    longest_job_company, longest_job_role, longest_job_duration = get_longest_time_at_a_job_duration_company(json_data)
    analysis_dict["Longest Time at a Job (Duration)"] = longest_job_duration
    analysis_dict["Longest Time at a Job (Company)"] = longest_job_company
    analysis_dict["Longest Time at a Job (Job Role)"] = longest_job_role

    # Get shortest job details
    shortest_job_company, shortest_job_role, shortest_job_duration = get_shortest_time_at_a_job_duration_company(json_data)
    analysis_dict["Shortest Time at a Job (Duration)"] = shortest_job_duration
    analysis_dict["Shortest Time at a Job (Company)"] = shortest_job_company
    analysis_dict["Shortest Time at a Job (Job Role)"] = shortest_job_role

    # Get more than 3 years experience details
    avg_duration_gt_3yrs, num_jobs_gt_3yrs, has_worked_gt_3yrs = has_worked_more_than_3_years(json_data)
    analysis_dict["Has worked for more than 3 years"] = has_worked_gt_3yrs
    analysis_dict["Total Number of Jobs worked for more than 3 years"] = num_jobs_gt_3yrs
    analysis_dict["Average Duration of Jobs worked for more than 3 years"] = avg_duration_gt_3yrs

    # Get less than 3 years experience details
    avg_duration_lt_3yrs, num_jobs_lt_3yrs, has_worked_lt_3yrs = has_worked_less_than_3_years(json_data)
    analysis_dict["Has worked for less than 3 years"] = has_worked_lt_3yrs
    analysis_dict["Total Number of Jobs worked for less than 3 years"] = num_jobs_lt_3yrs
    analysis_dict["Average Duration of Jobs worked for less than 3 years"] = avg_duration_lt_3yrs

    return analysis_dict

def is_job_hopper(analysis_dict, job_hopper_average_month_threshold,
                   job_hopper_average_month_less_than_3year_threshold,
                   job_hopper_total_jobs_less_than_3year_threshold, 
                   criteria_method):
    """
    Determines if an applicant is a job hopper based on the provided criteria.

    Args:
        analysis_dict: The dictionary containing the analyzed resume data.
        job_hopper_average_month_threshold: Threshold for average job duration (all jobs).
        job_hopper_average_month_less_than_3year_threshold: Threshold for average job duration (jobs less than 3 years).
        job_hopper_total_jobs_less_than_3year_threshold: Threshold for total number of jobs (less than 3 years).
        criteria_method: How to apply the criteria - "All must be true" or "At least one must be true".

    Returns:
        1 if the applicant is flagged as a job hopper, 0 otherwise.
    """

    # Condition checks, safely handling None values
    condition1 = job_hopper_average_month_threshold is not None and analysis_dict["Average Time at Jobs"] <= job_hopper_average_month_threshold
    condition2 = job_hopper_average_month_less_than_3year_threshold is not None and analysis_dict.get("Average Duration of Jobs worked for less than 3 years", float('inf')) <= job_hopper_average_month_less_than_3year_threshold
    condition3 = job_hopper_total_jobs_less_than_3year_threshold is not None and analysis_dict.get("Total Number of Jobs worked for less than 3 years", -1) >= job_hopper_total_jobs_less_than_3year_threshold
    
    if criteria_method == 0: # All criteria must be met
        is_job_hopper = all([condition1, condition2, condition3])
    elif criteria_method == 1: # At least one criterion must be met
        is_job_hopper = any([condition1, condition2, condition3])
    elif criteria_method == 2: # Only consider 'Average Time at Jobs'
        is_job_hopper = condition1
    elif criteria_method == 3: # Only consider 'Avg. Time at Jobs UNDER 3 Years'
        is_job_hopper = condition2
    elif criteria_method == 4: # Only consider 'Total Jobs UNDER 3 Years'
        is_job_hopper = condition3
    else:
        raise ValueError("Invalid criteria method.")

    return is_job_hopper

#For Job Hopper Resume Analysis: END

def job_fit_response(uploaded_file: UploadFile, job_description, tech_keywords, exp_keywords, edu_keywords, tech_weight, exp_weight, edu_weight):

    text = extract_text(uploaded_file.file)
    prompt = f"""Please evaluate the following resume based on the job description and criteria provided. Score the applicant from the range of (0-100) based on the criteria given.
    The weight will be only be used as future basis for hiring the applicant, but you can use this as a guide for how critical a criteria should be scored by you.\n\n
    Resume text: {text}\n\nJob Description: {job_description}\n\nHere are the weights of each of the criteria and the main keywords to look for:\n"""
                    
    tech_keywords_formatted = "\n".join(tech_keywords)
    exp_keywords_formatted = "\n".join(exp_keywords)
    edu_keywords_formatted = "\n".join(edu_keywords)

    # Now, use these variables directly in the f-string without backslashes in the expressions
    prompt += f"""
    1) Technical Skills: (Weight: {tech_weight}, Scoring: (0-100))
    Main Keywords for Evaluation:
    {tech_keywords_formatted}

    2) Relevant Experience: (Weight: {exp_weight}, Scoring: (0-100))
    Main Keywords for Evaluation:
    {exp_keywords_formatted}

    3) Education Relevance: (Weight: {edu_weight}, Scoring: (0-100))
    Main Keywords for Evaluation:
    {edu_keywords_formatted}
    """

    prompt += f"""\n
    ------------------------
    Provide your answer and scores in a json format following the structure below:
    {{
        "Name": <applicant's name>,
        "Technical Skills": <score (number only)>,
        "Relevant Experience": <score (number only)>,
        "Education Relevance": <score (number only)>,
        "Overall Score":<score (0-100) (number only)>,
        "Overall Assessment":<a text description or report on the overall assessment of the applicant>
    }}
    ------------------------\n
    """

    # Extract text from the PDF
    if uploaded_file is not None:

        # Choose a model that's appropriate for your use case.
        configure_genai()
        model = genai.GenerativeModel('gemini-1.5-flash',
            generation_config=genai.GenerationConfig(
            max_output_tokens=5000,
            temperature=0.4,
            response_mime_type = "application/json"
        ))

        response_json_valid = False
        is_expected_json = False
        max_attempts = 3
        parsed_result = {}

        while not response_json_valid and max_attempts > 0: ## Wait till the response json is valid
            response = ""
    
            try:
                
                response = model.generate_content(prompt).text
            except Exception as e:
                max_attempts = max_attempts - 1 
                print(f"Failed to process the following file: {uploaded_file.filename}...\n\n Due to error: f{str(e)}.\n\n Trying again... Retries left: {max_attempts} attempt/s")
                continue #Continue on next operations, this will skip the while loop entirely

            parsed_result, response_json_valid = extract_and_parse_json(response)
            if response_json_valid == False:
                print(f"Failed to validate and parse json for {uploaded_file.filename}... Trying again...")
                max_attempts = max_attempts - 1
                continue

            is_expected_json = is_expected_json_content(parsed_result, type = "job_fit")
            if is_expected_json == False:
                print(f"Successfully validated and parse json for {uploaded_file.filename} but is not expected format... Trying again...")
                print(f"Please review results: {parsed_result}\n\n")
                continue

            print(f"Parsed Results for {uploaded_file.filename}: {parsed_result}")

        if max_attempts == 0 and response_json_valid == False and is_expected_json == False:
            raise HTTPException(status_code=400, detail="Failed to process the document successfully. Please check the uploaded document.")

    return parsed_result


def generate_response(uploaded_file):
    primary_data_extract_query = f"""
Given the following resume data. I'd like you to extract only the important data points:
1) Name
2) Year Graduated (include the university or college name if available)
4) Job Histories (include company name and the duration)
    - Extract the starting dates and end dates
    - Then manually compute the exact durations. Be accurate with your computations, consider the month and the year.
    - Extract Job Role
    - Do not include or consider internships as part of job history

Notes:
- No yapping!
- If the resume text contains "Present" date, consider that today's date is {datetime.datetime.now().strftime("%B %d, %Y")}.

----------------------------------------
Resume Text:

    """

    query_text = f"""
Given the resume text below, I want you to extract me the following information/data from the resume in a json format:

{{
"Name":<name>,
"Year Graduated": <year>,
"Company History" : [
    {{
        "Company Name": <company name>,
        "Year Started" <year (number format only)>,
        "Job Role": <job role>,
        "Working Months": <total months working (number format only)>
    }}
    ]
}}

Note:
- Please strictly follow the json structure above.
- If the information doesn't exist enter a json "null" value.
- If no company history then pass an empty list.
----------------------------------------

Resume Information:

    """
    request_count = 0

    # Extract text from the PDF
    if uploaded_file is not None:

        # Choose a model that's appropriate for your use case.
        configure_genai()
        model = genai.GenerativeModel('gemini-1.0-pro',
            generation_config=genai.GenerationConfig(
            max_output_tokens=5000,
            temperature=0.4
        ))

        text = extract_text(uploaded_file)
        primary_prompt = primary_data_extract_query + text
        chat = model.start_chat(history=[])
        primary_response = None
        main_response = None

        if request_count <= 14:
            primary_response = chat.send_message(primary_prompt)
            # print(f"Primary Response Text:\n{primary_response.text}")
            request_count += 1
        else:
            # Raise an exception when the request limit for primary requests is reached
            raise Exception("Request limit reached for initial processing. Please wait a minute before trying again.")

        if request_count <= 15:
            main_prompt = query_text + primary_response.text 
            main_response = chat.send_message(main_prompt)
            # print(f"Main Response Text:\n{main_response.text}")
            request_count += 1
        else:
            # Raise an exception when the request limit for main requests is reached
            raise Exception("Request limit reached for detailed processing. Please wait a minute before trying again.")
        
    return main_response.text


#General Functions: END Here

@app.get("/docs")
def read_docs():
    return get_swagger_ui_html(openapi_url="/openapi.json")

@app.post("/job-hopper-identifier/", response_description="Contains the analysis of the job history, the job history JSON, and processing information about the document.")
async def job_hopper_identifier(
    file: UploadFile = File(..., description="Upload a PDF or DOC file containing the resume."),
    average_time_at_jobs_threshold: Optional[int] = Form(None, description="Threshold for average duration at jobs, used to determine job hopping."),
    average_time_under_3_years_threshold: Optional[int] = Form(None, description="Threshold for jobs lasting less than three years."),
    total_jobs_under_3_years_threshold: Optional[int] = Form(None, description="Number of jobs lasting less than three years required to qualify as a job hopper."),
    criteria_method: int = Form(..., description="Method to apply criteria: 0 - All criteria must be met, 1 - At least one criterion must be met, 2 - Only consider 'Average Time at Jobs', 3 - Only consider 'Avg. Time at Jobs UNDER 3 Years', 4 - Only consider 'Total Jobs UNDER 3 Years'")
    ):
    """
    Determine if an applicant is a job hopper based on their resume.

    Args:\n
        file (UploadFile): A PDF or DOCX resume file.\n
        average_time_at_jobs_threshold (int, optional): Average time spent at jobs to consider for job hopping.\n
        average_time_under_3_years_threshold (int, optional): Specific threshold for jobs under three years.\n
        total_jobs_under_3_years_threshold (int, optional): Total jobs under three years to qualify as job hopping.\n
        criteria_method (int): Specifies the method to determine job hopper status.\n
            0 - All criteria must be met.\n
            1 - At least one criterion must be met.\n
            2 - Only consider 'Average Time at Jobs'.\n
            3 - Only consider 'Avg. Time at Jobs UNDER 3 Years'.\n
            4 - Only consider 'Total Jobs UNDER 3 Years'.\n

    Returns:\n
        JobHopperResponse: Contains the analysis of the job history, the job history and information about the document processing.
    
    Raises:\n
        HTTPException: If the file format is not supported, or if the necessary thresholds are not provided for the selected criteria method.\n
        HTTPException: If any of the numerical thresholds provided are negative.\n
        HTTPException: If the criteria method is not within the valid range (0-4).\n
        HTTPException: If the document cannot be processed successfully, or if the JSON structure of the processed data is incorrect or unexpected.\n
        HTTPException: If an unexpected error occurs during the processing of the file.\n
    """
    
    #Error and Data validation: START
    print(f"Average Time at Jobs Threshold: {average_time_at_jobs_threshold}")
    print(f"Average Time Under 3 Years Threshold: {average_time_under_3_years_threshold}")
    print(f"Total Jobs Under 3 Years Threshold: {total_jobs_under_3_years_threshold}")
    print(f"Criteria Method: {criteria_method}")


    # Check if the file is in PDF or DOC format
    if file.content_type not in ["application/pdf", "application/msword", 
                                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="File format not supported. Only PDF or DOC files are accepted.")

    if criteria_method == 0:
        # Check if all criteria values are provided and are valid
        if any(x is None for x in [average_time_at_jobs_threshold, average_time_under_3_years_threshold, total_jobs_under_3_years_threshold]):
            raise HTTPException(status_code=400, detail="Incomplete criteria information. Please input a value for all 3 main criterias.")

        if any(x < 0 for x in [average_time_at_jobs_threshold, average_time_under_3_years_threshold, total_jobs_under_3_years_threshold]):
            raise HTTPException(status_code=400, detail="Criteria values must be non-negative.")

    # Check that at least one criterion is provided when criteria_method is 1 (at least one criterion must be met)
    elif criteria_method == 1:
        if all(criterion is None for criterion in [
            average_time_at_jobs_threshold,
            average_time_under_3_years_threshold,
            total_jobs_under_3_years_threshold
        ]):
            raise HTTPException(status_code=400, detail="At least one criterion must be provided for this method.")
    elif criteria_method == 2 and average_time_at_jobs_threshold is None:
        raise HTTPException(status_code=400, detail="Average time at jobs threshold must be provided.")
    elif criteria_method == 3 and average_time_under_3_years_threshold is None:
        raise HTTPException(status_code=400, detail="Average time under 3 years threshold must be provided.")
    elif criteria_method == 4 and total_jobs_under_3_years_threshold is None:
        raise HTTPException(status_code=400, detail="Total jobs under 3 years threshold must be provided.")

    #Error and Data validation: START
    try:
        response_json_valid = False
        is_expected_json = False
        max_attempts = 3
        parsed_result = {}

        # Process the file here (e.g., parsing, analyzing, etc.)
        while not response_json_valid and max_attempts > 0: ## Wait till the response json is valid
            analysis_result = ""

            #Test 1
            try:
                analysis_result = generate_response(file.file)
            except Exception as e:
                print(f"Failed to process the folloiwing file: {file.filename}...\n\n Due to error: f{str(e)}")
                max_attempts = max_attempts - 1 
                continue

            #Test 2
            parsed_result, response_json_valid = extract_and_parse_json(analysis_result)
            if response_json_valid == False:
                print(f"Failed to validate and parse json for {file.filename}... Trying again...")
                max_attempts = max_attempts - 1
                continue

            #Test 3
            is_expected_json = is_expected_json_content(parsed_result)
            if is_expected_json == False:
                print(f"Successfully validated and parse json for {file.filename} but is not expected format... Trying again...")
                continue

            print(f"Parsed Results for {file.filename}: {parsed_result}")
            analysis_dict = analyze_resume(parsed_result)
            analysis_dict["Is Job Hopper?"] = is_job_hopper(
                    analysis_dict,
                    average_time_at_jobs_threshold,
                    average_time_under_3_years_threshold,
                    total_jobs_under_3_years_threshold,
                    criteria_method,
                )
            
            return{
                "applicant_job_history_analysis": analysis_dict,
                "applicant_job_history_json":parsed_result,
                "processing_info": f"'{file.filename}' was processed successfully under {2-max_attempts} attempt/s only."
            }

        if max_attempts == 0 and response_json_valid == False and  is_expected_json == False:
            raise HTTPException(status_code=400, detail="Failed to process the document successfully. Please check the uploaded document.")
 

#       return {"info": f"file '{file.filename}' was processed and the temporary file has been removed."}
    except Exception as e:
        # Handle unexpected errors
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/job-fit-identifier/", response_description="Evaluates a resume against specific job criteria and returns the analysis.")
async def job_fit_identifier(
    file: UploadFile = File(..., description="Upload a PDF or DOC file containing the resume."),
    job_description: str = Form(..., description="The detailed job description against which the resume will be evaluated."),
    tech_keywords: List[str] = Form(..., description="List of technical skill keywords to be matched in the resume."),
    exp_keywords: List[str] = Form(..., description="List of experience-related keywords to check in the resume."),
    edu_keywords: List[str] = Form(..., description="List of educational background keywords to assess in the resume."),
    tech_weight: int = Form(..., description="Weight (0-100) given to technical skills in the overall evaluation."),
    exp_weight: int = Form(..., description="Weight (0-100) for experience relevance in the overall assessment."),
    edu_weight: int = Form(..., description="Weight (0-100) for educational background relevance in the job fit evaluation.")
):
    """
    This endpoint processes the uploaded resume and evaluates it against the provided job description using the specified criteria.\n
    It calculates scores for technical skills, relevant experience, and educational relevance based on the weights and keywords provided.\n
    \n
    The function returns a structured JSON containing scores and a textual assessment of how well the resume matches the job description.\n
    \n
    Args:\n
        file (UploadFile): The resume file to be analyzed.\n
        job_description (str): A string detailing the job requirements and responsibilities.\n
        tech_keywords (List[str]): Keywords related to the technical skills required for the job.\n
        exp_keywords (List[str]): Keywords associated with the experience required for the job.\n
        edu_keywords (List[str]): Keywords linked to the educational qualifications necessary for the job.\n
        tech_weight (int): Numerical weight indicating the importance of technical skills in the overall job fit.\n
        exp_weight (int): Numerical weight indicating the importance of relevant experience in the overall job fit.\n
        edu_weight (int): Numerical weight indicating the importance of educational background in the job fit evaluation.\n
    \n
    Returns:\n
        dict: A dictionary containing the evaluation scores and an overall assessment of the resume's fit for the job.\n
    \n
    Raises:\n
        HTTPException: If the file format is not supported, or if any of the weights are outside the acceptable range (0-100).\n
    """

    # Validate weights
    weight_fields = [tech_weight, exp_weight, edu_weight]
    if any(w < 0 or w > 100 for w in weight_fields):
        raise HTTPException(status_code=400, detail="Weights must be non-negative and not exceed 100.")

    if file.content_type not in ["application/pdf", "application/msword", 
                                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="File format not supported. Only PDF or DOC files are accepted.")
    
    # Call the function to process the resume and compute the job fit
    try:
        results = job_fit_response(
            uploaded_file=file,
            job_description=job_description,
            tech_keywords=tech_keywords,
            exp_keywords=exp_keywords,
            edu_keywords=edu_keywords,
            tech_weight=tech_weight,
            exp_weight=exp_weight,
            edu_weight=edu_weight
        )
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while processing the file: {str(e)}")

# Endpoint to process a PDF or DOC file and extract specified data points
@app.post("/resume-data-extraction/", response_description="Extracts specified data points from an uploaded resume.")
async def resume_data_extraction(
    file: UploadFile = File(..., description="Upload a PDF, DOC, or DOCX file containing the resume."),
    data_points: List[str] = Form(..., description="A list of data points to extract from the resume, such as 'Name', 'Education', 'Work History'.")
):
    """
    This endpoint extracts specified information from a provided resume. The user must define what specific data points should be extracted, 
    and the system will attempt to find and return these data points in a structured JSON format.\n
    \n
    Args:\n
        file (UploadFile): The resume file to be analyzed. Supported file types are PDF, DOC, and DOCX.\n
        data_points (List[str]): A list of strings that specifies which pieces of information to extract from the resume. Examples include 'Name', 'Contact Information', 'Work Experience', etc.\n
    \n
    Returns:\n
        dict: A dictionary where each key corresponds to a requested data point and the value is the extracted information or 'NA' if the information does not exist.\n
    \n
    Raises:\n
        HTTPException: If the file format is not supported.\n
        HTTPException: If the document cannot be processed or if the extraction fails to produce valid JSON.\n
        HTTPException: If an unexpected error occurs during the processing.\n
    """

    # Validate the file type
    if file.content_type not in ["application/pdf", "application/msword", 
                                 "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
        raise HTTPException(status_code=400, detail="File format not supported. Only PDF, DOC, or DOCX files are accepted.")

    # Read and extract text from the uploaded file
    text = extract_text(file.file)

    # Prepare the extraction prompt
    # prompt = f"""Resume text:\n{text}\n\nExtract the following information from the resume, enter NA if it doesn't exist:\n\n"""
    # for point in data_points:
    #     prompt += f"{point}\n"
    # prompt += f"""\n
    # ------------------------
    # Provide your answer in a json format following the structure below:
    # {{
    # """ 
    # for idx, point in enumerate(data_points):
    #     prompt += f"\"{point}\": <information>"
    #     if idx < len(data_points) - 1:  # Check if it's not the last item
    #         prompt += ","
    #     prompt += "\n"
    # prompt += f"""}}
    # --------------
    # """

    # print("Variable Type: ", type(data_points))
    # print(len(data_points))
    ### Sometime the data_points are passed as a list containing only one string for all data points but concatenated with comma
    parsed_data_points = [item.strip() for sublist in data_points for item in sublist.split(',')]
    prompt = f"""Resume text:\n{text}\n\nExtract the following information from the resume, enter NA if it doesn't exist:\n\n"""
    for point in parsed_data_points:
        prompt += f"{point}\n"
        print(point)
    prompt += f"""\n
    ------------------------
    Provide your answer in a json format following the structure below:
    {{
    """ 
    for idx, point in enumerate(parsed_data_points):
        prompt += f"\"{point}\": <information>"
        if idx < len(parsed_data_points) - 1:  # Check if it's not the last item
            prompt += ","
        prompt += "\n"
    prompt += f"""}}
    --------------
    """
    
    # Call the generative AI model to process the extraction prompt
    print("The prompt: ", prompt)
    configure_genai()
    model = genai.GenerativeModel('gemini-1.5-flash',  # Adjust model and configuration as needed
                                  generation_config=genai.GenerationConfig(
                                      max_output_tokens=3000,
                                      temperature=0.4,
                                      response_mime_type="application/json"))
    
    # Generate the content based on the prompt
    try:
        response_json_valid = False
        max_attempts = 3
        parsed_result = {}

        while not response_json_valid and max_attempts>0: ## Wait till the response json is valid
            response = model.generate_content(prompt,
                            safety_settings={
                            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                        }).text
            parsed_result, response_json_valid = extract_and_parse_json(response)
            print("Result: ", parsed_result)
            if response_json_valid == False:
                print(f"Failed to validate and parse json for {file.filename}... Trying again...")
                max_attempts = max_attempts - 1 

        if max_attempts == 0 and response_json_valid == False:
            raise HTTPException(status_code=400, detail="Failed to process the document successfully. Please check the uploaded document and you data points requests")
        else:
            print("Result: ", parsed_result)
            return {"results": parsed_result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# # This function is optional for this setup but can be used for running the server
if __name__ == "__main__":
    configure_genai() #configure genai on first run
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
