from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Crear app Flask
app = Flask(__name__)
CORS(app)  # Habilita CORS para peticiones externas

# Cargar el modelo generador de diálogo
print("→ Cargando modelo generador (DialoGPT)...")
tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

@app.route("/responder", methods=["POST"])
def responder():
    data = request.get_json()

    if not data or "mensaje" not in data:
        return jsonify({"error": "Falta el campo 'mensaje'"}), 400

    mensaje = data["mensaje"].strip()

    if not mensaje:
        return jsonify({"error": "El mensaje está vacío"}), 400

    # Tokenizar entrada del usuario
    input_ids = tokenizer.encode(mensaje + tokenizer.eos_token, return_tensors="pt")

    # Generar respuesta con el modelo
    respuesta_ids = model.generate(input_ids, max_length=100, pad_token_id=tokenizer.eos_token_id)

    # Decodificar respuesta generada
    respuesta = tokenizer.decode(respuesta_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return jsonify({"respuesta": respuesta})

if __name__ == "__main__":
    app.run(debug=True)
