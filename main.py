import os
import uuid
import base64
import shutil
import threading
import urllib.parse
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

# ── Referência de composição (calibrada para mundosenai.png 1080×1920) ────────
_REF_W,      _REF_H      = 1080, 1920
_REF_X,      _REF_Y_BASE =   74,  220
_REF_FOTO_W, _REF_FOTO_H =  900,  550
_REF_BORDA               =   10

# ── Config por estação (nome do evento, QR overlay URL, modo LGPD) ───────────
_station_config: dict[str, dict] = {}
_cfg_lock = threading.Lock()

def get_station_config(station_id: str) -> dict:
    with _cfg_lock:
        if station_id not in _station_config:
            _station_config[station_id] = {
                'event_name': 'SENAI — Totem de Fotos',
                'overlay_qr_url': '',
                'overlay_qr_label': '',
                'lgpd_mode': 'personal',   # 'personal' | 'promotional'
            }
        return dict(_station_config[station_id])

def update_station_config(station_id: str, **kwargs):
    with _cfg_lock:
        cfg = _station_config.setdefault(station_id, {
            'event_name': 'SENAI — Totem de Fotos',
            'overlay_qr_url': '',
            'overlay_qr_label': '',
            'lgpd_mode': 'personal',
        })
        cfg.update(kwargs)

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
                'photo_id':       None,
                'cloudinary_url': None,
                'foto_capturada': False,
                'espera':         False,
            }
        return _sessions[sid]


# ── Registro de fotos (photo_id → metadados) ──────────────────────────────────
_photo_registry: dict[str, dict] = {}
_reg_lock = threading.Lock()

def registrar_foto(photo_id: str, cloudinary_url: str | None, station_id: str, event_name: str):
    with _reg_lock:
        _photo_registry[photo_id] = {
            'cloudinary_url': cloudinary_url,
            'station_id':     station_id,
            'event_name':     event_name,
        }

def get_foto_info(photo_id: str) -> dict | None:
    with _reg_lock:
        return _photo_registry.get(photo_id)

def get_foto_cloudinary_url(photo_id: str) -> str | None:
    """Constrói URL Cloudinary a partir do photo_id quando o registro em memória expirou."""
    if not CLOUDINARY_CONFIGURED:
        return None
    info = get_foto_info(photo_id)
    if info and info.get('cloudinary_url'):
        return info['cloudinary_url']
    # Tenta buscar direto no Cloudinary com public_id previsível
    try:
        resource = cloudinary.api.resource(f'cabinefotos/fotos/{photo_id}')
        return resource['secure_url']
    except Exception:
        return None


# ── Cloudinary upload de fotos ────────────────────────────────────────────────
def upload_foto_cloudinary(filepath: str, photo_id: str, event_name: str = '') -> str | None:
    if not CLOUDINARY_CONFIGURED:
        return None
    try:
        result = cloudinary.uploader.upload(
            filepath,
            public_id=f'cabinefotos/fotos/{photo_id}',
            overwrite=False,
            resource_type='image',
            context=f'event_name={event_name}',
        )
        return result['secure_url']
    except Exception as e:
        print(f" * Cloudinary foto upload error: {e}")
        return None


# ── QR overlay na imagem composta ─────────────────────────────────────────────
_SENAI_RED_BGR = (19, 6, 227)   # #E30613 em BGR

def aplicar_qr_overlay(moldura: np.ndarray, url: str, label: str,
                       bg_w: int, bg_h: int) -> np.ndarray:
    """QR com borda vermelha SENAI, fundo branco e faixa de texto personalizável."""
    try:
        qr_size = max(80, int(bg_w * 0.09))    # tamanho do QR em pixels
        pad     = max(7,  int(qr_size * 0.09)) # espaço branco ao redor do QR
        border  = max(5,  int(qr_size * 0.06)) # largura da borda vermelha
        margin  = max(12, int(bg_w * 0.013))   # distância da borda da imagem

        # Gera o QR (preto no branco)
        qr_img = qrcode.make(url)
        qr_arr = np.array(qr_img.convert('RGB'))
        qr_bgr = cv2.cvtColor(qr_arr, cv2.COLOR_RGB2BGR)
        qr_bgr = cv2.resize(qr_bgr, (qr_size, qr_size), interpolation=cv2.INTER_AREA)

        # Área branca com QR + borda vermelha
        inner   = qr_size + 2 * pad
        panel_w = inner + 2 * border

        # Faixa de texto abaixo (só se label preenchido)
        font    = cv2.FONT_HERSHEY_DUPLEX
        fscale  = max(0.28, qr_size / 300)
        fthick  = max(1, int(qr_size / 72))
        text    = label.upper() if label else ''
        (tw, th), baseline = cv2.getTextSize(text, font, fscale, fthick) if text else ((0, 0), 0)
        label_h = (th + baseline + max(10, int(qr_size * 0.12))) if text else 0

        panel_h = inner + 2 * border + label_h
        panel   = np.full((panel_h, panel_w, 3), _SENAI_RED_BGR, dtype=np.uint8)

        # Área branca central
        panel[border : border + inner, border : border + inner] = (255, 255, 255)

        # QR no centro da área branca
        panel[border + pad : border + pad + qr_size,
              border + pad : border + pad + qr_size] = qr_bgr

        # Texto na faixa vermelha inferior
        if text:
            tx = max(border, (panel_w - tw) // 2)
            ty = border + inner + border + th + max(4, int(qr_size * 0.04))
            cv2.putText(panel, text, (tx, ty), font, fscale,
                        (255, 255, 255), fthick, cv2.LINE_AA)

        # Cola no canto inferior direito da moldura
        px = bg_w - panel_w - margin
        py = bg_h - panel_h - margin
        if py >= 0 and px >= 0:
            moldura[py : py + panel_h, px : px + panel_w] = panel

    except Exception as e:
        print(f" * QR overlay error: {e}")
    return moldura


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


@app.route('/station_info')
def station_info():
    station_id = request.args.get('station', 'default')
    return jsonify(get_station_config(station_id))


@app.route('/config_estacao', methods=['POST'])
def config_estacao():
    station_id = request.form.get('station_id', 'default')
    update_station_config(
        station_id,
        event_name=request.form.get('event_name', '').strip() or 'SENAI — Totem de Fotos',
        overlay_qr_url=request.form.get('overlay_qr_url', '').strip(),
        overlay_qr_label=request.form.get('overlay_qr_label', '').strip(),
        lgpd_mode=request.form.get('lgpd_mode', 'personal'),
    )
    return jsonify(success=True)


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

    # Salva cada foto no tamanho de referência; o composite redimensiona para o fundo atual
    frame    = cv2.resize(frame, (_REF_FOTO_W, _REF_FOTO_H))
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

        # Escala as coordenadas proporcionalmente às dimensões do fundo atual
        bg_h, bg_w = moldura.shape[:2]
        sx = bg_w / _REF_W
        sy = bg_h / _REF_H
        x      = int(_REF_X      * sx)
        y_base = int(_REF_Y_BASE * sy)
        w      = int(_REF_FOTO_W * sx)
        h      = int(_REF_FOTO_H * sy)
        borda  = max(4, int(_REF_BORDA * sy))

        nome_comp = f'foto_{station_id[:8]}_{uuid.uuid4()}.png'
        photo_id  = nome_comp[:-4]
        bg_foto   = np.zeros((h + borda, w + borda, 3), dtype=np.uint8)

        y = y_base
        for seq in range(3):
            foto = cv2.imread(os.path.join(PHOTOS_DIR, f'{sid_pfx}_foto_{seq}.png'))
            if foto is None:
                continue
            foto = cv2.resize(foto, (w, h))
            moldura[y - borda//2 : y + h + borda//2,
                    x - borda//2 : x + w + borda//2] = bg_foto
            moldura[y : y + h, x : x + w] = foto
            y += h + max(10, int(15 * sy))

        # QR overlay no canto inferior direito (fora da área das fotos)
        cfg         = get_station_config(station_id)
        overlay_url   = cfg.get('overlay_qr_url', '')
        overlay_label = cfg.get('overlay_qr_label', '')
        if overlay_url:
            moldura = aplicar_qr_overlay(moldura, overlay_url, overlay_label, bg_w, bg_h)

        comp_path  = os.path.join(PHOTOS_DIR, nome_comp)
        cv2.imwrite(comp_path, moldura)

        event_name = cfg.get('event_name', 'SENAI — Totem de Fotos')
        cloud_url  = upload_foto_cloudinary(comp_path, photo_id, event_name)
        registrar_foto(photo_id, cloud_url, station_id, event_name)

        sess['nome_foto']      = nome_comp
        sess['photo_id']       = photo_id
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
    photo_id  = sess.get('photo_id')

    if cloud_url:
        qr_data = cloud_url
    elif photo_id:
        qr_data = f'{APP_URL}foto/{photo_id}'
    elif sess.get('nome_foto'):
        qr_data = f'{APP_URL}download/{sess["nome_foto"]}'
    else:
        return Response(status=404)

    img       = qrcode.make(qr_data)
    img_bytes = BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    return Response(img_bytes.read(), mimetype='image/png')


@app.route('/foto/<photo_id>')
def download_page(photo_id):
    info       = get_foto_info(photo_id)
    cloud_url  = get_foto_cloudinary_url(photo_id)
    event_name = (info or {}).get('event_name', 'SENAI — Totem de Fotos')

    if not cloud_url:
        local_path = os.path.join(PHOTOS_DIR, photo_id + '.png')
        if not os.path.exists(local_path):
            return render_template('download.html',
                error=True, photo_id=photo_id, event_name=event_name,
                cloud_url=None, whatsapp_url=None)

    whatsapp_url = None
    if cloud_url:
        msg = urllib.parse.quote(f'Confira minha foto! {cloud_url}')
        whatsapp_url = f'https://api.whatsapp.com/send?text={msg}'

    return render_template('download.html',
        photo_id=photo_id,
        cloud_url=cloud_url,
        event_name=event_name,
        whatsapp_url=whatsapp_url,
        error=False,
    )


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
