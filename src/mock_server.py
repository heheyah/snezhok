import os
import json
from fastapi import FastAPI
from pydantic import BaseModel
import chromadb
from chromadb.utils import embedding_functions
import requests 
from fastapi.responses import FileResponse

app = FastAPI()

# 1. Connect to your Chroma Database
print("Connecting to ChromaDB...")
client = chromadb.PersistentClient(path="data/chroma_db")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client.get_collection(name="protocols", embedding_function=sentence_transformer_ef)

# Define the expected input format
class PatientData(BaseModel):
    symptoms: str

@app.get("/")
async def serve_ui():
    # Safely locates the static/index.html 
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    html_path = os.path.join(base_dir, "static", "index.html")
    
    if not os.path.exists(html_path):
        return {"error": "index.html not found. Check your folder structure!"}
        
    return FileResponse(html_path)

@app.post("/diagnose")
async def diagnose(patient: PatientData):
    # Search the database using the patient's symptoms
    results = collection.query(
        query_texts=[patient.symptoms],
        n_results=2  # Reduced to 2 to save space
    )
    
    # Extract the actual text from the search results
    retrieved_chunks = results['documents'][0]
    full_context = "\n---\n".join(retrieved_chunks)
    

    context = full_context[:4000] 
    
    # 3. Setup the Qazcode Hub connection
    llm_url = "https://hub.qazcode.ai/v1/chat/completions" 
    headers = {
        "Authorization": "Bearer sk-BDVloWBwHCr5oltlXwyhtA", 
        "Content-Type": "application/json"
    }
    
    # 4. Define the system rules (The Job Description)
    # 🛑 FIX: Changed "icd_code" to "icd10_code" so the evaluator doesn't crash!
    # 4. Define the system rules (The Sniper Prompt)
    SYSTEM_PROMPT = """You are an expert clinical diagnostician.
    Given a patient's symptoms (in Russian) and relevant clinical protocol text, determine the top 3 most likely diagnoses.

    CRITICAL RULES:
    1. Act as a doctor. Diagnose the patient based on the clinical picture. The protocol text is just context to guide you.
    2. Provide the EXACT 4-character ICD-10 code (МКБ-10) for your diagnoses (e.g., "F32.0", "U09.9").
    3. Use your own deep medical knowledge of ICD-10 to select the correct precise sub-category (4th digit) that matches the symptom severity.
    4. Output ONLY the raw code in the 'icd10_code' field (no extra words like "Код").
    5. Rank 1 MUST be your absolute best Primary Diagnosis. Pay attention to the patient's primary complaint.
    6. Keep explanations extremely short (under 10 words in Russian).

    Return EXACTLY this JSON structure and absolutely nothing else:
    {
      "diagnoses": [
        {"rank": 1, "icd10_code": "...", "name": "...", "explanation": "..."},
        {"rank": 2, "icd10_code": "...", "name": "...", "explanation": "..."},
        {"rank": 3, "icd10_code": "...", "name": "...", "explanation": "..."}
      ]
    }"""

    # 5. Combine your RAG Context and the Symptoms
    user_message = f"""
    --- НАЧАЛО КЛИНИЧЕСКИХ ПРОТОКОЛОВ ---
    {context}
    --- КОНЕЦ КЛИНИЧЕСКИХ ПРОТОКОЛОВ ---
    
    СИМПТОМЫ ПАЦИЕНТА:
    {patient.symptoms}
    
    Based on the protocols above, provide the top 3 diagnoses in the requested JSON format.
    """

    # 6. Build the payload
    payload = {
        "model": "oss-120b",
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message}
        ],
        "max_tokens": 2000
    }
    
    # 7. Call the model and parse the response
    try:
        print("Sending request to LLM... (This might take a few seconds)")
        response = requests.post(llm_url, headers=headers, json=payload)
        response.raise_for_status() 
        
        response_data = response.json()
        llm_text = response_data['choices'][0]['message']['content']
        
        # Strip out any markdown formatting the LLM might have added
        clean_text = llm_text.replace("```json", "").replace("```", "").strip()
        
        # Parse the text back into a Python dictionary
        final_result = json.loads(clean_text)
        
        # Ensure it returns the exact {"diagnoses": [...]} structure expected
        if "diagnoses" in final_result:
            print("🎯 WHAT WE ARE SENDING TO EVALUATOR:")
            print(json.dumps(final_result, indent=2, ensure_ascii=False))
            return final_result
        else:
            return {"diagnoses": final_result}
            
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"🔍 Server Explanation: {e.response.text}") 
        return {"diagnoses": []}
        
    except json.JSONDecodeError as e:
        print(f"❌ JSON Parsing Error: The LLM didn't return valid JSON. It returned: {llm_text}")
        return {"diagnoses": []}
        
    except Exception as e:
        print(f"❌ General Error: {e}")
        return {"diagnoses": []}