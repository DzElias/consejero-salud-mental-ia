from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import os

app = Flask(__name__)
CORS(app)

# Configurar tu clave secreta de OpenAI (¡segura!)
openai.api_key = "open-ai-api-key"

@app.route("/responder", methods=["POST"])
def responder():
    data = request.get_json()

    if not data or "mensaje" not in data:
        return jsonify({"error": "Falta el campo 'mensaje'"}), 400

    mensaje = data["mensaje"]

    try:
        respuesta = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Puedes cambiar a "gpt-4" si tienes acceso
            messages=[
                {"role": "system", "content": "Actúa como un amigo empático que da consejos útiles según el contexto."},
                {"role": "user", "content": mensaje}
            ],
            temperature=0.7,
            max_tokens=200
        )

        texto_respuesta = respuesta.choices[0].message["content"].strip()
        return jsonify({"respuesta": texto_respuesta})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
