from flask import Flask, jsonify, request
from flask_cors import CORS
import subprocess
import json
import os
import base64

app = Flask(__name__)
CORS(app)

# Ruta al entorno virtual
venv_path = "./venv"
python_path = os.path.join(venv_path, "Scripts", "python")
main_script_path = os.path.join(os.path.dirname(__file__), "main.py")
temp_image_path = os.path.join(venv_path, "cuadrado.jpg")  # Debe coincidir con main.py

@app.route('/check-camera', methods=['GET'])
def check_camera():
    """Endpoint para verificar que el backend funciona"""
    return jsonify({"status": "Backend Flask conectado correctamente"})

@app.route('/analyze-face', methods=['POST'])
def analyze_face():
    """Endpoint para ejecutar el análisis facial con imagen capturada"""
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                "success": False,
                "error": "No se proporcionó imagen",
                "message": "Imagen requerida para el análisis"
            }), 400

        # Obtener la imagen base64 del frontend
        image_base64 = data['image']

        # Guardar la imagen temporalmente
        try:
            image_bytes = base64.b64decode(image_base64.split(',')[-1])
            with open(temp_image_path, 'wb') as f:
                f.write(image_bytes)
            print(f">>> Imagen guardada en: {temp_image_path}")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Error al guardar la imagen: {str(e)}",
                "message": "No se pudo procesar la imagen enviada"
            }), 400

        # Ejecutar el script main.py con la ruta de la imagen como parámetro
        result = subprocess.run([
            python_path,
            main_script_path,
            temp_image_path  # Pasar la ruta de la imagen como parámetro
        ], capture_output=True, text=True, timeout=30, encoding='utf-8')

        print(f">>> Resultado del script: {result.returncode}")

        if result.returncode == 0:
            # Buscar el JSON en la salida
            lines = result.stdout.strip().split('\n')
            json_line = None

            for line in reversed(lines):
                line = line.strip()
                if line.startswith('{') and line.endswith('}'):
                    try:
                        json.loads(line)
                        json_line = line
                        break
                    except Exception:
                        continue

            if not json_line:
                json_line = lines[-1] if lines else ""

            print(f">>> JSON encontrado: {json_line[:100]}...")

            try:
                analysis_result = json.loads(json_line)
                return jsonify({
                    "success": True,
                    "data": analysis_result,
                    "message": "Análisis completado exitosamente"
                })
            except json.JSONDecodeError as e:
                print(f">>> Error decodificando JSON: {e}")
                return jsonify({
                    "success": False,
                    "error": "Formato de respuesta inválido del script Python",
                    "stdout_preview": result.stdout[:200] + "..." if len(result.stdout) > 200 else result.stdout,
                    "stderr_preview": result.stderr[:200] + "..." if result.stderr and len(result.stderr) > 200 else result.stderr
                }), 500
        else:
            return jsonify({
                "success": False,
                "error": result.stderr,
                "stdout": result.stdout,
                "message": "Error en el análisis facial"
            }), 500

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "El análisis tardó demasiado tiempo",
            "message": "Timeout del análisis"
        }), 500
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Error interno del servidor"
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Endpoint para verificar el estado del servidor"""
    return jsonify({
        "status": "healthy", 
        "service": "OptiScan Backend",
        "python_path": python_path,
        "script_exists": os.path.exists(main_script_path),
        "venv_exists": os.path.exists(venv_path)
    })

if __name__ == '__main__':
    print(">>> Iniciando servidor Flask para OptiScan...")
    print(f">>> Python path: {python_path}")
    print(f">>> Script path: {main_script_path}")
    print(f">>> Script existe: {os.path.exists(main_script_path)}")
    print(f">>> Venv existe: {os.path.exists(venv_path)}")
    
    app.run(debug=True, port=5000, host='0.0.0.0')