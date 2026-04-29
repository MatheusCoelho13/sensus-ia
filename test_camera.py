import cv2
import requests
import io

URL = "http://127.0.0.1:8000/analisar"

def capture_and_send():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Não foi possível abrir a câmera")
        return

    ret, frame = cap.read()
    cap.release()

    if not ret:
        print("Falha ao capturar frame")
        return

    # encode to JPEG
    success, img_buf = cv2.imencode('.jpg', frame)
    if not success:
        print("Falha ao codificar imagem")
        return

    files = {'file': ('frame.jpg', img_buf.tobytes(), 'image/jpeg')}
    try:
        resp = requests.post(URL, files=files, timeout=30)
        print('Status:', resp.status_code)
        try:
            print(resp.json())
        except Exception:
            print(resp.text)
    except Exception as e:
        print('Erro na requisição:', e)

if __name__ == '__main__':
    capture_and_send()
