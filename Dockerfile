FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure KBFILES folder exists with clinical KB documents
# These are the 7 peer-reviewed clinical reference files
RUN mkdir -p KBFILES medagent_kb_docs

# Set KB_DIR so nodes.py loads from KBFILES
ENV KB_DIR=./KBFILES

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]