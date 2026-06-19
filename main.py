import os
import uuid
import base64
import threading

import cv2
import numpy as np
import qrcode
import cloudinary
import cloudinary.uploader
from flask import Flask, render_template, send_file, Response, request, jsonify, redirect
from io import BytesIO

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PHOTOS_DIR = os.path.join(app.root_path, 'static', 'photos')
FUNDO_PATH = os.path.join(app.root_path, 'static', 'images', 'mundosenai.png')
os.makedirs(PHOTOS_DIR, exist_ok=True)

# ── Configuração via variáveis de ambiente ────────────────────────────────────
APP_URL = os.environ.get('APP_URL', 'http://localhost:2205/')

CLOUDINARY_CONFIGURED = bool(os.environ.get('CLOUDINARY_CLOUD_NAME'))
if CLOUDINARY_CONFIGURED:
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
        secure=True
    )
    print(f" * Cloudinary configurado: {os.environ.get('CLOUDINARY_CLOUD_NAME')}")
else:
    print(" * Cloudinary NÃO configurado — usando armazenamento local")

# ── Estado por sessão ─────────────────────────────────────────────────────────
_sessions: dict = {}
_sessions_lock  = threading.Lock()

def get_session(sid: str) -> dict:
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = {
                'foto_seq':       0,
                'nome_foto':      '',
                'cloudinary_url': None,
                'foto_capturada': False,
                'espera':         False,
            }
        return _sessions[sid]


# ── Cloudinary ────────────────────────────────────────────────────────────────
def upload_cloudinary(filepath: str) -> str | None:
    """Faz upload da imagem composta e retorna a URL pública segura."""
    if not CLOUDINARY_CONFIGURED:
        return None
    try:
        result = cloudinary.uploader.upload(
            filepath,
            folder='cabinefotos',
            resource_type='image',
        )
        return result['secure_url']
    except Exception as e:
        print(f" * Cloudinary upload error: {e}")
        return None


# ── Rotas ─────────────────────────────────────────────────────────────────────

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/configuracao', methods=['GET', 'POST'])
def configuracao():
    global APP_URL
    msg = None
    if request.method == 'POST':
        url = request.form.get('site_url', '').strip()
        if url:
            APP_URL = url if url.endswith('/') else url + '/'
            msg = ('success', f'URL atualizada: {APP_URL}')

    return render_template(
        'configuracao.html',
        app_url=APP_URL,
        cloudinary_ok=CLOUDINARY_CONFIGURED,
        cloudinary_cloud=os.environ.get('CLOUDINARY_CLOUD_NAME', '—'),
        msg=msg,
    )


@app.route('/upload_foto', methods=['POST'])
def upload_foto():
    sid  = request.args.get('sid', 'default')
    sess = get_session(sid)

    x, y_base, w, h, borda = 74, 220, 900, 550, 10

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify(success=False, error='No image data'), 400

    img_data = data['image']
    if ',' in img_data:
        img_data = img_data.split(',')[1]

    nparr = np.frombuffer(base64.b64decode(img_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify(success=False, error='Invalid image'), 400

    frame   = cv2.resize(frame, (w, h))
    sid_pfx = sid[:8]
    seq_name = f'{sid_pfx}_foto_{sess["foto_seq"]}.png'
    cv2.imwrite(os.path.join(PHOTOS_DIR, seq_name), frame)
    sess['nome_foto'] = seq_name
    sess['foto_seq'] += 1

    if sess['foto_seq'] == 3:
        moldura   = cv2.imread(FUNDO_PATH)
        nome_comp = f'foto_{uuid.uuid4()}.png'
        bg_foto   = np.zeros((h + borda, w + borda, 3), dtype=np.uint8)

        y = y_base
        for seq in range(3):
            foto = cv2.imread(os.path.join(PHOTOS_DIR, f'{sid_pfx}_foto_{seq}.png'))
            moldura[y - borda//2 : y + h + borda//2,
                    x - borda//2 : x + w + borda//2] = bg_foto
            moldura[y : y + h, x : x + w] = foto
            y += h + 15

        comp_path = os.path.join(PHOTOS_DIR, nome_comp)
        cv2.imwrite(comp_path, moldura)

        # Upload para Cloudinary (assíncrono não necessário — arquivo é pequeno)
        cloud_url = upload_cloudinary(comp_path)

        sess['nome_foto']       = nome_comp
        sess['cloudinary_url']  = cloud_url
        sess['foto_capturada']  = True
        sess['foto_seq']        = 0
        sess['espera']          = True

    return jsonify(success=True, foto_seq=sess['foto_seq'])


@app.route('/captured_image')
def captured_image():
    sid  = request.args.get('sid', 'default')
    sess = get_session(sid)

    # Cloudinary disponível → redireciona para a URL pública
    if sess.get('cloudinary_url'):
        return redirect(sess['cloudinary_url'])

    path = os.path.join(PHOTOS_DIR, sess['nome_foto'])
    return send_file(path, mimetype='image/png')


@app.route('/captured')
def captured():
    sid  = request.args.get('sid', 'default')
    sess = get_session(sid)
    fotocap               = sess['foto_capturada']
    sess['foto_capturada'] = False
    return jsonify(foto_capturada=fotocap)


@app.route('/qr')
def qr():
    sid  = request.args.get('sid', 'default')
    sess = get_session(sid)

    # QR aponta direto para o Cloudinary (sem passar pelo servidor)
    cloud_url = sess.get('cloudinary_url')
    if cloud_url:
        qr_data = cloud_url
    elif sess.get('nome_foto'):
        qr_data = f'{APP_URL}download/{sess["nome_foto"]}'
    else:
        return Response(status=404)

    img       = qrcode.make(qr_data)
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return Response(img_bytes.read(), mimetype='image/png')


@app.route('/download/<filename>')
def download_image(filename):
    path = os.path.join(PHOTOS_DIR, filename)
    return send_file(path, as_attachment=True)


@app.route('/esperadetect')
def esperadetect():
    sid  = request.args.get('sid', 'default')
    sess = get_session(sid)
    sess['espera'] = False
    return 'ok'


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 2205))
    app.run(host='0.0.0.0', port=port, threaded=True)
