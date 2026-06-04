import requests
import json
import os

BASE_URL = "http://localhost:8000"

def run_test():
    print("--- RAG SYSTEM INTEGRATION TEST ---")
    
    # 1. Check health/settings
    print("\n1. Checking Settings/Health API...")
    res = requests.get(f"{BASE_URL}/api/settings")
    print(f"Status Code: {res.status_code}")
    print(f"Response: {res.text}")
    assert res.status_code == 200, "Settings API failed"
    
    # 2. Upload a sample customer concern document
    print("\n2. Uploading a sample concern document...")
    sample_text = (
        "Customer Concern Log - June 2026\n"
        "ID: CONCERN-101\n"
        "Product: SmartHome Hub v2\n"
        "Issue: The device suddenly drops Wi-Fi connection every day at 14:00 PM. "
        "The light on the front flashes red for two minutes, and then it reconnects. "
        "This disrupts office workflows.\n\n"
        "ID: CONCERN-102\n"
        "Product: EcoPower Battery Pack\n"
        "Issue: The battery pack gets extremely hot during fast charging. "
        "The temperature rises above 55 degrees Celsius. "
        "The charging automatically slows down as a safety mechanism, but the shell remains hot to touch.\n"
    )
    
    temp_filename = "sample_concerns.txt"
    with open(temp_filename, "w", encoding="utf-8") as f:
        f.write(sample_text)
        
    try:
        with open(temp_filename, "rb") as f:
            files = {"file": (temp_filename, f, "text/plain")}
            data = {"chunk_size": 150, "chunk_overlap": 20}
            res = requests.post(f"{BASE_URL}/api/upload", files=files, data=data)
            
        print(f"Status Code: {res.status_code}")
        print(f"Response: {res.text}")
        assert res.status_code == 200, "Upload failed"
        upload_data = res.json()
        doc_id = upload_data["doc_id"]
        print(f"Successfully indexed document with ID: {doc_id}")
        
    finally:
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
    # 3. List documents to verify
    print("\n3. Listing active indexed documents...")
    res = requests.get(f"{BASE_URL}/api/documents")
    print(f"Response: {res.text}")
    assert len(res.json()) > 0, "Document list is empty"
    
    # 4. Perform a semantic retrieval query
    print("\n4. Performing a search query on the indexed concerns...")
    query_payload = {
        "query": "Which device drops its Wi-Fi connection and flashes red?",
        "top_k": 2,
        "search_mode": "hybrid",
        "hybrid_weight": 0.5,
        "provider": "mock" # Offline mode
    }
    
    res = requests.post(f"{BASE_URL}/api/query", json=query_payload)
    print(f"Status Code: {res.status_code}")
    print("Response JSON:")
    print(json.dumps(res.json(), indent=2))
    
    assert res.status_code == 200, "Query API failed"
    response_data = res.json()
    assert len(response_data["contexts"]) > 0, "No contexts retrieved!"
    print("\n--- TEST PASSED SUCCESSFULLY! ---")

if __name__ == "__main__":
    try:
        run_test()
    except Exception as e:
        print(f"\n--- TEST FAILED! ---")
        print(e)
