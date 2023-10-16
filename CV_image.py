import fitz  # need to do (pip install PyMuPDF) first
from PIL import Image
import io 

# Define the PDF file, image file, and output file names
pdf_file = 'CVS_Di_Marcantonio_no_image.pdf'
image_file = "C:/Users/HP/Downloads/Photo_Di_Marcantonio.png"
output_file = 'CVS_Di_Marcantonio.pdf'

# Load the PDF document
pdf_document = fitz.open(pdf_file)

# Load the image using PIL
image = Image.open(image_file)

# Get the first page of the PDF document
page = pdf_document[0]

# Get the size of the page
page_width = page.mediabox_size[0]
page_height = page.mediabox_size[1]

# Calculate the position and size of the image
x = 20  # Adjust the X-coordinate as needed
y = 7  # Adjust the Y-coordinate as needed
image_width = 100  # Adjust the width as needed
image_height = image_width * (image.height / image.width)  # Maintain aspect ratio

# Convert the image to a BytesIO object
image_bytes = io.BytesIO()
image.save(image_bytes, format='PNG')
image_data = image_bytes.getvalue()

# Add the image to the page
image_rect = fitz.Rect(x, y, x + image_width, y + image_height)
page.insert_image(image_rect, stream=image_data)

# Save the modified PDF to the output file
pdf_document.save(output_file)

# Close the PDF document
pdf_document.close()
