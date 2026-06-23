from pypdf import PdfReader

def extract_text_from_resume(file_path):
    """Extracts raw text from a resume PDF, page by page."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    return full_text

if __name__ == "__main__":
    text = extract_text_from_resume("data/sample_resume.pdf")
    print(text)
    print("\n--- Character count:", len(text), "---")