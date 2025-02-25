import os
import fitz  # PyMuPDF for PDF processing
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader


class PDFProcessor:
    """
    Class for processing PDF files.

    Attributes:
        file: The uploaded PDF file (e.g., Streamlit UploadedFile object).
        temp_pdf_path: Path to the temporary saved PDF file.
        images: List of file paths for images extracted from PDF pages.
        full_text: The extracted text content from the PDF.
    """

    def __init__(self, file):
        self.file = file
        self.temp_pdf_path = None
        self.images = []
        self.full_text = None

    def save_temp_file(self):
        """
        Save the uploaded PDF to a temporary file.

        Returns:
            The file path of the temporary PDF.
        """
        temp_path = "temp.pdf"
        with open(temp_path, "wb") as f:
            f.write(self.file.getvalue())
        self.temp_pdf_path = temp_path
        return self.temp_pdf_path

    def extract_images(self):
        """
        Convert each page of the PDF into an image and store the paths.

        Returns:
            List of image file paths.
        """
        if not self.temp_pdf_path:
            self.save_temp_file()
        doc = fitz.open(self.temp_pdf_path)
        image_paths = []
        os.makedirs("static", exist_ok=True)
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap()
            img_path = os.path.join("static", f"page_{i + 1}.png")
            pix.save(img_path)
            image_paths.append(img_path)
        self.images = image_paths
        return self.images

    def combine_images(self, max_pages=None):
        """
        Vertically combine multiple images into a single image.

        Args:
            max_pages: If specified, only combine the first max_pages images.

        Returns:
            A combined PIL Image object or None if no images are available.
        """
        if not self.images:
            self.extract_images()
        imgs = [Image.open(path) for path in self.images[:max_pages] if os.path.exists(path)]
        if not imgs:
            return None
        widths, heights = zip(*(img.size for img in imgs))
        max_width = max(widths)
        total_height = sum(heights)
        combined_image = Image.new("RGB", (max_width, total_height))
        y_offset = 0
        for img in imgs:
            combined_image.paste(img, (0, y_offset))
            y_offset += img.size[1]
        return combined_image

    def extract_text(self):
        """
        Extract text from the PDF using PyPDFLoader.

        Returns:
            A string containing the concatenated text of the PDF.
        """
        if not self.temp_pdf_path:
            self.save_temp_file()
        loader = PyPDFLoader(self.temp_pdf_path)
        docs = loader.load()
        self.full_text = "\n".join(doc.page_content for doc in docs)
        return self.full_text
