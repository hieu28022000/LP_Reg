import pytesseract
from PIL import Image

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def Get_text(img_path):
    img = Image.open(img_path)
    # img = img.convert('L')
    # Image._show(img)
    text = pytesseract.image_to_string(img)
    print(text)
    return text, img_path

if __name__ == '__main__':
    text, image_path = Get_text('License_Plates\License_Plates.jpg')
    print(len(text))