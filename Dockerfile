# 1. Basis-Image mit Python
FROM python:3.11-slim

# 2. Arbeitsverzeichnis
WORKDIR /app

# 3. Abhängigkeiten kopieren und installieren
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 4. Anwendungscode kopieren
COPY . .

# 5. Umgebungsvariable für Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 6. Port freigeben
EXPOSE 5000

# 7. Startbefehl
CMD ["flask", "run"]
