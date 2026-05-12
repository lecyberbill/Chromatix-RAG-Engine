import requests
import json

url = "http://localhost:8000/ask"
payload = {"query": "Définition des ambiances intérieures et atmosphères extérieures"}

print(f"--- [TEST] Envoi de la requête au Backend... ---")
try:
    response = requests.post(url, json=payload, stream=True)
    print(f"--- [TEST] Réponse reçue (Streaming) : ---")
    for line in response.iter_lines():
        if line:
            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:].strip()
                if data_str == "[DONE]":
                    break
                data = json.loads(data_str)
                if "chunk" in data:
                    print(data["chunk"], end="", flush=True)
                elif "error" in data:
                    print(f"\n!!! Erreur API : {data['error']}")
    print("\n--- [TEST] Fin du stream ---")
except Exception as e:
    print(f"\n!!! Erreur de connexion : {e}")
