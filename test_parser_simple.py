import os
from src.parser import extract_text_from_file

# Path to your sample resume
file_path = r"C:\Users\hp\resume-check-mvp\data\resume.pdf"

# Check if file exists
print("File exists?", os.path.exists(file_path))

# Open the file in binary mode
with open(file_path, "rb") as f:
    # Attach the MIME type attribute directly
    f.type = "application/pdf"  # change to docx MIME if needed
    text = extract_text_from_file(f)

# Print preview
print("----- Extracted Text Preview -----")
if text:
    print(text[:200])  # first 200 chars
else:
    print("⚠️ No text extracted. Check the file format or content.")
print("----- End Preview -----")

