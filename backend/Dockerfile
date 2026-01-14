FROM python:3.11-slim

WORKDIR /app

# התקנת תלויות
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# העתקת קבצי הקוד
COPY . .

EXPOSE 8000

