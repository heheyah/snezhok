# ❄️ Team Snezhok - Clinical Diagnostic AI (Docker Submission)

This project is a Clinical Decision Support System designed to match patient symptoms to official medical protocols using Retrieval-Augmented Generation (RAG). 

## 🚀 Build and Run Instructions

### 1. Build the Docker Image
Navigate to the root directory of the project (where the `Dockerfile` is located) and run the following command to build the image:

```bash
docker build -t submission .
```
### 2. Run the container
Once built, start the container on port 8080:
```bash
docker run -p 8080:8080 submission
```
## 🖥️ Usage
### Web UI
Open your web browser and go to:
#### 👉 http://localhost:8080/
You can enter a patient's free-text symptoms directly into the interface to receive the top 3 diagnoses and ICD-10 codes.

### API Endpoint (POST)
You can send POST requests to the inference endpoint directly:
#### 👉 http://localhost:8080/diagnose
### Example request:
```JSON
{
  "symptoms": "Пациент жалуется на сильную головную боль, ухудшение зрения и постоянную жажду."
}
```
## 🏛️ System Architecture
* Single Container Deployment: The app uses a unified FastAPI server serving both the Web UI and the JSON API on port 8080.\
* Vector Database: Pre-indexed document chunks from corpus.zip are baked directly into the Docker image, requiring no external database connections.
* Offline Inference: All RAG generation and symptom matching are done entirely within the container without reaching out to external internet APIs.
* Prompt Engineering: Optimized to prioritize specific 4-character ICD-10 codes explicitly stated in the retrieved protocols over generic code families.
***
