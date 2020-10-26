"""utility to add metadata 
"""
from PIL import Image
import piexif

zeroth_ifd = {
              piexif.ImageIFD.Make: u"Autoencoder",
              piexif.ImageIFD.Software: u"piexif"
              }

exif_dict = {"0th":zeroth_ifd}
exif_bytes = piexif.dump(exif_dict)
im = Image.open("/home/felix/Desktop/Document_Scanner/images/output.jpg")
im.save("output.jpg", exif=exif_bytes)

img = Image.open('/home/felix/Desktop/Document_Scanner/output.jpg')
exif_data = img._getexif()
print(exif_data)