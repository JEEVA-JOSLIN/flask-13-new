apt-get update
apt-get install -y libmagic1
apt-get install -y build-essential
apt-get install -y python3-dev
apt-get install -y libjpeg-dev
apt-get install -y libmupdf-dev mupdf-tools
apt-get install -y tesseract-ocr 
apt-get install -y libtesseract-dev 
apt-get install -y poppler-utils 
apt-get install -y libgl1-mesa-glx 
apt-get install -y libglib2.0-0
apt-get install -y libopencv-dev
apt-get install -y libatlas-base-dev 
apt-get install -y libopenjp2-7 
apt-get install -y libtiff5
gunicorn --workers 3 --bind 0.0.0.0:8000 app:app --log-level debug
