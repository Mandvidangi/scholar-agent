import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path):
    """
    Extract text from all pages of a PDF file.
    """

    document = fitz.open(pdf_path)
    full_text = ""

    for page_number in range(len(document)):
        page = document[page_number]

        # Extract text from current page
        text = page.get_text()

        # Add page number so later we can cite page source
        full_text += f"\n\n--- Page {page_number + 1} ---\n"
        full_text += text

    document.close()

    return full_text