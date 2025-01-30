import re
import os
import pandas as pd
from PyPDF2 import PdfReader
from docx import Document
from tqdm import tqdm
from typing import Dict, List

def extract_text(file_path: str) -> str:
    """Extract text from PDF or DOCX files"""
    try:
        if file_path.endswith('.pdf'):
            return extract_text_from_pdf(file_path)
        elif file_path.endswith('.docx'):
            return extract_text_from_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error extracting text from {file_path}: {str(e)}")

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF using PyPDF2"""
    text = []
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        for page in reader.pages:
            text.append(page.extract_text() or '')
    return ' '.join(text)

def extract_text_from_docx(docx_path: str) -> str:
    """Extract text from DOCX using python-docx"""
    doc = Document(docx_path)
    return ' '.join([paragraph.text for paragraph in doc.paragraphs])

def clean_text(text: str) -> str:
    """Clean and preprocess extracted text"""
    text = re.sub(r'[^\w\s.,-]', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)         # Remove extra whitespace
    return text.strip().lower()

def parse_llm_output(output: str, section: str) -> Dict:
    """Parse LLM output into structured data"""
    parsed = {}
    try:
        if section == 'personal_info':
            for line in output.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    parsed[key.strip().lower()] = value.strip()
        
        elif section == 'work_experience':
            parsed["experience"] = []
            jobs = output.split('Job ')
            for job in jobs[1:]:  # Skip empty first element
                job_data = {}
                lines = job.split('\n')
                job_data["title"] = lines[0].split(':', 1)[1].strip()
                for line in lines[1:]:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        job_data[key.strip().lower()] = value.strip()
                parsed["experience"].append(job_data)
        
        elif section == 'skills':
            parsed["skills"] = [skill.strip() for skill in output.split(',')]
        
        elif section == 'education':
            parsed["education"] = []
            entries = output.split('\n')
            for entry in entries:
                if any(keyword in entry.lower() for keyword in ['degree', 'institution', 'graduation', 'gpa']):
                    parsed["education"].append(entry.strip())
        
        elif section == 'projects':
            parsed["projects"] = []
            projects = output.split('Project ')
            for proj in projects[1:]:  # Skip empty first element
                proj_data = {}
                lines = proj.split('\n')
                proj_data["title"] = lines[0].split(':', 1)[1].strip()
                for line in lines[1:]:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        proj_data[key.strip().lower()] = value.strip()
                parsed["projects"].append(proj_data)
        
        return parsed
    
    except Exception as e:
        raise RuntimeError(f"Error parsing {section} section: {str(e)}")

def process_resumes(input_dir: str, output_dir: str, chains: Dict) -> None:
    """Process all resumes in input directory and save results to single CSV"""
    os.makedirs(output_dir, exist_ok=True)
    resumes = [f for f in os.listdir(input_dir) if f.endswith(('.pdf', '.docx'))]
    all_parsed_data = []
    
    for resume_file in tqdm(resumes, desc="Processing Resumes"):
        try:
            file_path = os.path.join(input_dir, resume_file)
            raw_text = extract_text(file_path)
            cleaned_text = clean_text(raw_text)
            
            # Extract features using LLM chains
            features = {
                'personal_info': chains['personal_info'].run({'text': cleaned_text}),
                'professional_summary': chains['professional_summary'].run({'text': cleaned_text}),
                'work_experience': chains['work_experience'].run({'text': cleaned_text}),
                'skills': chains['skills'].run({'text': cleaned_text}),
                'education': chains['education'].run({'text': cleaned_text}),
                'projects': chains['projects'].run({'text': cleaned_text}),
            }
            
            # Parse all sections
            parsed_data = {}
            for section, output in features.items():
                parsed_data.update(parse_llm_output(output, section))
            
            # Add filename and raw text for reference
            parsed_data['source_file'] = resume_file
            parsed_data['raw_text'] = cleaned_text[:500] + "..."  # Store first 500 chars
            
            all_parsed_data.append(parsed_data)
            
        except Exception as e:
            print(f"\nError processing {resume_file}: {str(e)}")
            continue
    
    if all_parsed_data:
        # Create combined DataFrame
        combined_df = pd.DataFrame(all_parsed_data)
        
        # Save to single CSV file
        output_path = os.path.join(output_dir, "all_parsed_resumes.csv")
        combined_df.to_csv(output_path, index=False)
        print(f"\nSuccessfully processed {len(all_parsed_data)} resumes. Saved to {output_path}")
    else:
        print("\nNo resumes were successfully processed")