import librosa
from flask import Flask, request
from transformers import pipeline

app = Flask(__name__)

@app.route('/procesar-audio', methods=['POST'])
def procesar_audio():
    # Verifica que se haya enviado un archivo de audio en la solicitud
    if 'audio' not in request.files:
        return 'No se ha enviado ningún archivo de audio', 400
    
    archivo_audio = request.files['audio']
    
    # Realiza aquí el procesamiento del archivo de audio
    response = obtain_audio_information(archivo_audio)
    
    # Retorna la respuesta del procesamiento del audio
    return response

def obtain_audio_information(archivo_audio):
    print('Audio information')
    #utilizamos la biblioteca librosa  para obtener la informacion del audio
    audio, frecuencia_muestreo = librosa.load(archivo_audio)
    # Retorna la duración del audio en segundos
    duration = len(audio) / frecuencia_muestreo

    #obtenemos los modelos sentiment-analysis y automatic-speech-recognition
    Clasificador = pipeline('sentiment-analysis')
    transcripcion = pipeline('automatic-speech-recognition')

    #Realizamos la transcripción y el analisis de sentimientos 
    text_process = transcripcion(audio)['text']
    clasification_res = Clasificador(text_process)
        
    response = {
        "status": 200,
        "data": {
            "duration": f'Duración del audio: {duration} segundos',
            "text_process": text_process,
            "clasification": clasification_res
        }
    }
    return response

if __name__ == '__main__':
    app.run()
