import streamlit as st
import easyocr
import re
import nltk
from PIL import Image
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
import numpy as np

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

def extract_information(ocr_text):
    # Initialize variables to store extracted information
    emails = []
    person_name = ""
    company_name = ""
    address = ""
    phone_number = ""
    desg = ""
    website = ""

    # Tokenize the OCR text
    words = word_tokenize(ocr_text)

    # Perform Part-of-Speech tagging
    tagged_words = pos_tag(words)

    # Extract emails using regular expressions
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    emails = re.findall(email_pattern, ocr_text)

    # Extract person's name and organization name using NLTK's named entity recognition (NER)
    named_entities = nltk.ne_chunk(tagged_words)
    for chunk in named_entities:
        if isinstance(chunk, nltk.Tree):
            if chunk.label() == 'PERSON':
                person_name = ' '.join([token[0] for token in chunk if token[1] != 'POS'])
            elif chunk.label() == 'ORGANIZATION':
                company_name = ' '.join([token[0] for token in chunk if token[1] != 'POS'])

    # Extract designation using provided logic
    desg_tokens = []
    for word in words:
        if word.lower() in ["ceo", "founder", "inc", "llc", "ltd"]:
            desg_tokens.append(word)

    if desg_tokens:
        desg = " ".join(desg_tokens)

    # Extract address using a custom heuristic
    address_tokens = []
    start_collecting = False
    for word, pos in tagged_words:
        if pos == 'NNP' or pos == 'CD' and word != person_name and word != company_name: # Proper noun or cardinal number, potentially part of an address
            address_tokens.append(word)

    # Join address tokens into a single string
    address = ' '.join(address_tokens)

    # Extract phone numbers using regular expressions
    phone_pattern = r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'
    phone_numbers = re.findall(phone_pattern, ocr_text)
    if phone_numbers:
        phone_number = phone_numbers[0]

    # Extract website using regular expressions
    website_pattern = r'\b(?:https?://|www\.)\S+\b'
    websites = re.findall(website_pattern, ocr_text)
    if websites:
        website = websites[0]

    return emails, person_name, company_name, address.strip(), phone_number, desg, website

# Create Streamlit interface
def ocr_extraction(image):
    try:
        # Create the reader object with desired languages
        reader = easyocr.Reader(['en'], gpu=False)

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Perform OCR on the image
        text_read = reader.readtext(image_array)

        # Extract the OCR text from the result
        ocr_text = ' '.join([res[1] for res in text_read])

        # Extract information from OCR text
        #ocr_text = ocr_text.lower()
        emails, person_name, company_name, address, phone_number, desg, website = extract_information(ocr_text)

        return {
            "Emails": emails,
            "Person's Name": person_name,
            "Company Name": company_name,
            #"Address": address,
            "Phone Number": phone_number,
            "Designation": desg,
            "Website": website
        }
    except:
        return "Error: Failed to process the image. Please try again with a different image."

# Streamlit interface components
st.title("Bizcard")
st.write("Upload an image of a business card to extract information.")
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Extracted Information:")
    result = ocr_extraction(image)
    for key, value in result.items():
        if value == "":
          continue
        st.write(f"**{key}:** {value}")
