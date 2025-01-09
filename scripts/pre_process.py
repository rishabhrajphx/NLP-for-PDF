import fitz  # PyMuPDF
import json
import os
import re
import usaddress  # for better address parsing
from typing import Dict, List, Tuple

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract text from PDF while preserving formatting"""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        # Extract text while preserving line breaks
        text += page.get_text("text")
    return text.strip()

def clean_text(text: str) -> str:
    """
    Clean text while preserving important information
    """
    # Normalize spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove unwanted special characters but keep important punctuation
    text = re.sub(r'[^\w\s@.,-]', '', text)
    return text.strip()

def extract_names(text: str) -> Tuple[str, str]:
    """
    Extract first and last names using improved pattern matching
    Handles middle names, hyphens, and other formats
    """
    name_pattern = r'\b([A-Z][a-z]+(?:[- ][A-Z][a-z]+)*)\b'
    names = re.findall(name_pattern, text)
    if names:
        return names[0], names[1] if len(names) > 1 else ""
    return "", ""

def extract_address(text: str) -> str:
    """
    Extract address using usaddress library for better parsing
    """
    try:
        # Look for common address patterns first
        address_pattern = r'\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd)[,\s]+[A-Za-z\s]+,?\s*[A-Z]{2}\s*\d{5}'
        match = re.search(address_pattern, text)
        if match:
            return match.group()
        
        # Try usaddress parsing as backup
        parsed_address = usaddress.tag(text)
        if parsed_address:
            return parsed_address[0]
    except:
        pass
    return ""

def extract_university(text: str) -> str:
    """
    Extract university name using improved pattern matching
    """
    university_patterns = [
        r'\b(?:University|College|Institute)\s+of\s+[A-Za-z\s]+\b',
        r'\b[A-Za-z]+\s+(?:University|College|Institute)\b',
        r'\b[A-Za-z]+\s+(?:University|College|Institute)\s+of\s+[A-Za-z\s]+\b'
    ]
    
    for pattern in university_patterns:
        match = re.search(pattern, text)
        if match:
            return match.group()
    return ""

def extract_entities(text: str) -> Dict[str, Tuple[int, int]]:
    """
    Extract entities using improved pattern matching
    """
    entities = {}
    
    # Extract names
    first_name, last_name = extract_names(text)
    if first_name:
        entities['First Name'] = (text.find(first_name), text.find(first_name) + len(first_name))
    if last_name:
        entities['Last Name'] = (text.find(last_name), text.find(last_name) + len(last_name))
    
    # Extract email
    email_match = re.search(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', text)
    if email_match:
        entities['Email'] = (email_match.start(), email_match.end())
    
    # Extract phone with various formats
    phone_patterns = [
        r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
        r'\d{3}[-.\s]\d{3}[-.\s]\d{4}',
        r'\+\d{1,2}[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
    ]
    for pattern in phone_patterns:
        phone_match = re.search(pattern, text)
        if phone_match:
            entities['Phone Number'] = (phone_match.start(), phone_match.end())
            break
    
    # Extract address
    address = extract_address(text)
    if address:
        addr_start = text.find(address)
        entities['Address'] = (addr_start, addr_start + len(address))
    
    # Extract university
    university = extract_university(text)
    if university:
        univ_start = text.find(university)
        entities['University'] = (univ_start, univ_start + len(university))
    
    # Extract GitHub profile
    github_match = re.search(r'github\.com/[\w-]+', text)
    if github_match:
        entities['Github Account'] = (github_match.start(), github_match.end())
    
    # Extract LinkedIn profile
    linkedin_match = re.search(r'linkedin\.com/in/[\w-]+', text)
    if linkedin_match:
        entities['Linkedin Account'] = (linkedin_match.start(), linkedin_match.end())
    
    return entities

def format_training_data(text: str, entities: Dict[str, Tuple[int, int]]) -> Dict:
    """Format extracted entities into training data format"""
    formatted_entities = []
    for label, (start, end) in entities.items():
        formatted_entities.append({
            "start": start,
            "end": end,
            "label": label
        })
    
    return {
        "text": text,
        "entities": formatted_entities
    }

def process_pdf_directory(pdf_dir: str, output_file: str):
    """Process PDFs and save results to JSON"""
    training_data = []
    
    # Process each PDF in the directory
    for filename in os.listdir(pdf_dir):
        if filename.endswith('.pdf'):
            
            # Extract and clean text
            pdf_path = os.path.join(pdf_dir, filename)
            
            # Format for training
            raw_text = extract_text_from_pdf(pdf_path)
            cleaned_text = clean_text(raw_text)
            entities = extract_entities(cleaned_text)
            training_example = format_training_data(cleaned_text, entities)
            training_data.append(training_example)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_data, f, indent=2)
    
    # Process PDFs and generate training data

if __name__ == "__main__":
    pdf_dir = "data/raw/"
    output_file = "data/processed/train_data.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    process_pdf_directory(pdf_dir, output_file)
    print(f"Processing complete. Results saved to {output_file}")