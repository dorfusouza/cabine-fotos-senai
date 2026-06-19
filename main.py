import os
import uuid
import base64
import shutil
import threading
import urllib.request

import cv2
import numpy as np
import qrcode
import cloudinary
import cloudinary.api
import cloudinary.uploader
from flask import Flask, render_template, send_file, Response, request, jsonify, redirect
from io import BytesIO

app = Flask(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
PHOTOS_DIR     = os.path.join(app.root_path, 'static', 'photos')
FUNDO_DEFAULT  = os.path.join(app.root_path, 'static', 'images', 'mundosenai.png')
FUNDO_CACHE_DIR = os.path.join(app.root_path, 'static', 'fundos')
ALLOWED_EXT    = {'png', 'jpg', 'jpeg'}
os.makedirs(PHOTOS_DIR,     exist_ok=True)
os.makedirs(FUNDO_CACHE_DIR, exist_ok=True)

# ── Configuração via variáveis de ambiente ────────────────────────────────────
APP_URL = os.environ.get('APP_URL', 'http://localhost:2205/')

CLOUDINARY_CONFIGURED = bool(os.environ.get('CLOUDINARY_CLOUD_NAME'))
if CLOUDINARY_CONFIGURED:
    cloudinary.config(
        cloud_name=os.environ.get('CLOUDINARY_CLOUD_NAME'),
        api_key=os.environ.get('CLOUDINARY_API_KEY'),
        api_secret=os.environ.get('CLOUDINARY_API_SECRET'),
        secure=True,
    )
    print(f" * Cloudinary configurado: {os.environ.get('CLOUDINARY_CLOUD_NAME')}")
else:
    print(" * Cloudinary NÃO configurado — usando armazenamento local")

# ── Fundos por estação ────────────────────────────────────────────────────────
_station_backgrounds: dict[str, str] = {}
_sbg_lock = threading.Lock()


def _station_cache_path(station_id: str) -> str:
    return os.path.join(FUNDO_CACHE_DIR, f'fundo_{station_id[:8]}.png')


def get_station_fundo(station_id: str) -> str:
    """Retorna o caminho do fundo desta estação.
    Busca em: memória → disco → Cloudinary → padrão."""
    with _sbg_lock:
        if station_id in _station_backgrounds:
            return _station_backgrounds[station_id]

    cache_path = _station_cache_path(station_id)
    if os.path.exists(cache_path):
        with _sbg_lock:
            _station_backgrounds[station_id] = cache_path
        return cache_path

    if CLOUDINARY_CONFIGURED:
        try:
            public_id = f'cabinefotos/fundo_{station_id[:8]}'
            resource  = cloudinary.api.resource(public_id)
            urllib.request.urlretrieve(resource['secure_url'], cache_path)
            with _sbg_lock:
                _station_backgrounds[station_id] = cache_path
            print(f" * Fundo estação {station_id[:8]} carregado do Cloudinary")
            return cache_path
        except Exception:
            pass

    return FUNDO_DEFAULT


def set_station_fundo(station_id: str, source_path: str) -> str:
    """Salva o fundo da estação localmente e sobe para o Cloudinary."""
    cache_path = _station_cache_path(station_id)
    shutil.copy(source_path, cache_path)
    with _sbg_lock:
        _station_backgrounds[station_id] = cache_path

    if CLOUDINARY_CONFIGURED:
        try:
            public_id = f'cabinefotos/fundo_{station_id[:8]}'
            cloudinary.uploader.upload(
                cache_path,
                public_id=public_id,
                overwrite=True,
                resource_type='image',
            )
        except Exception as e:
            print(f" * Cloudinary fundo upload error: {e}")

    return cache_path


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


# ── Cloudinary upload de fotos ────────────────────────────────────────────────
def upload_foto_cloudinary(filepath: str) -> str | None:
    if not CLOUDINARY_CONFIGURED:
        return None
    try:
        result = cloudinary.uploader.upload(
            filepath, folder='cabinefotos', resource_type='image'
        )
        return result['secure_url']
    except Exception as e:
        print(f" * Cloudinary foto upload error: {e}")
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


@app.route('/upload_fundo', methods=['POST'])
def upload_fundo():
    station_id = request.form.get('station_id', 'default')

    if 'fundo' not in request.files:
        return jsonify(success=False, error='Nenhum arquivo enviado'), 400

    file = request.files['fundo']
    ext  = file.filename.rsplit('.', 1)[-1].lower() if '.' in file.filename else ''
    if not ext or ext not in ALLOWED_EXT:
        return jsonify(success=False, error='Formato inválido. Use PNG ou JPG.'), 400

    tmp = os.path.join(FUNDO_CACHE_DIR, f'tmp_{uuid.uuid4()}.{ext}')
    try:
        file.save(tmp)
        if cv2.imread(tmp) is None:
            return jsonify(success=False, error='Arquivo inválido ou corrompido.'), 400
        set_station_fundo(station_id, tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    return jsonify(success=True)


@app.route('/fundo_preview')
def fundo_preview():
    station_id = request.args.get('station', 'default')
    path = get_station_fundo(station_id)
    return send_file(path, mimetype='image/png')


@app.route('/upload_foto', methods=['POST'])
def upload_foto():
    sid        = request.args.get('sid',     'default')
    station_id = request.args.get('station', 'default')
    sess       = get_session(sid)

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

    frame    = cv2.resize(frame, (w, h))
    sid_pfx  = sid[:8]
    seq_name = f'{sid_pfx}_foto_{sess["foto_seq"]}.png'
    cv2.imwrite(os.path.join(PHOTOS_DIR, seq_name), frame)
    sess['nome_foto'] = seq_name
    sess['foto_seq'] += 1

    if sess['foto_seq'] == 3:
        fundo_path = get_station_fundo(station_id)
        moldura    = cv2.imread(fundo_path)
        if moldura is None:
            moldura = cv2.imread(FUNDO_DEFAULT)
        if moldura is None:
            return jsonify(success=False, error='Fundo não pôde ser carregado'), 500

        nome_comp = f'foto_{uuid.uuid4()}.png'
        bg_foto   = np.zeros((h + borda, w + borda, 3), dtype=np.uint8)

        y = y_base
        for seq in range(3):
            foto = cv2.imread(os.path.join(PHOTOS_DIR, f'{sid_pfx}_foto_{seq}.png'))
            if foto is None:
                continue
            moldura[y - borda//2 : y + h + borda//2,
                    x - borda//2 : x + w + borda//2] = bg_foto
            moldura[y : y + h, x : x + w] = foto
            y += h + 15

        comp_path = os.path.join(PHOTOS_DIR, nome_comp)
        cv2.imwrite(comp_path, moldura)

        cloud_url = upload_foto_cloudinary(comp_path)

        sess['nome_foto']      = nome_comp
        sess['cloudinary_url'] = cloud_url
        sess['foto_capturada'] = True
        sess['foto_seq']       = 0
        sess['espera']         = True

    return jsonify(success=True, foto_seq=sess['foto_seq'])


@app.route('/captured_image')
def captured_image():
    sid  = request.args.get('sid', 'default')
    sess = get_session(sid)
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
