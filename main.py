import os
import uuid
import json
import base64
import shutil
import time
import secrets
import threading
import zipfile
import functools
import datetime
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict, field as dc_field

import cv2
import numpy as np
import qrcode
import cloudinary
import cloudinary.api
import cloudinary.uploader
from flask import Flask, render_template, send_file, Response, request, jsonify, redirect, session
from io import BytesIO
from PIL import Image

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

# ── Paths ─────────────────────────────────────────────────────────────────────
PHOTOS_DIR      = os.path.join(app.root_path, 'static', 'photos')
FUNDO_DEFAULT   = os.path.join(app.root_path, 'static', 'images', 'mundosenai.png')
FUNDO_DEFAULT_SINGLE = os.path.join(app.root_path, 'static', 'images', 'exposenai.png')
FUNDO_CACHE_DIR = os.path.join(app.root_path, 'static', 'fundos')
DATA_DIR        = os.path.join(app.root_path, 'data', 'events')
ALLOWED_EXT     = {'png', 'jpg', 'jpeg'}
os.makedirs(PHOTOS_DIR,      exist_ok=True)
os.makedirs(FUNDO_CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR,        exist_ok=True)

# ── Configuração via variáveis de ambiente ────────────────────────────────────
APP_URL        = os.environ.get('APP_URL', 'http://localhost:2205/')
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', '')

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

def _cloudinary_img_url(public_id: str, **kwargs) -> str:
    if not CLOUDINARY_CONFIGURED or not public_id:
        return ''
    try:
        return cloudinary.CloudinaryImage(public_id).build_url(**kwargs)
    except Exception:
        return ''

app.jinja_env.globals['cloudinary_img_url'] = _cloudinary_img_url

# ── Modelo de Evento ──────────────────────────────────────────────────────────
@dataclass
class Event:
    id:           str
    name:         str
    created_at:   str
    background_id: str = ''   # Cloudinary public_id ou nome de arquivo local
    qr_url:       str = ''
    qr_label:     str = ''
    lgpd_mode:    str = 'personal'
    active:       bool = True
    photo_count:  int  = 0
    capture_mode: str = 'strip'   # 'strip' (filipeta, 3 fotos) | 'single' (1 foto)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'Event':
        known = cls.__dataclass_fields__
        return cls(**{k: v for k, v in d.items() if k in known})


class EventStore:
    _CLOUD_PREFIX = 'cabinefotos/meta/events/'

    def __init__(self):
        self._cache: dict[str, Event] = {}
        self._lock  = threading.Lock()

    def _local_path(self, event_id: str) -> str:
        return os.path.join(DATA_DIR, f'{event_id}.json')

    @staticmethod
    def _strip_ext(eid: str) -> str:
        """Remove extensão .json que o Cloudinary raw pode acrescentar ao public_id."""
        return eid[:-5] if eid.endswith('.json') else eid

    def save(self, event: Event):
        os.makedirs(DATA_DIR, exist_ok=True)
        data = json.dumps(event.to_dict(), ensure_ascii=False, indent=2)
        with open(self._local_path(event.id), 'w', encoding='utf-8') as f:
            f.write(data)
        if CLOUDINARY_CONFIGURED:
            # Tmp SEM extensão: evita que Cloudinary raw inclua ".json" no public_id
            tmp = os.path.join(DATA_DIR, f'_tmp_{event.id}')
            try:
                with open(tmp, 'w', encoding='utf-8') as f:
                    f.write(data)
                cloudinary.uploader.upload(
                    tmp,
                    public_id=f'{self._CLOUD_PREFIX}{event.id}',
                    resource_type='raw', overwrite=True,
                )
            except Exception as e:
                print(f' * EventStore save: {e}')
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        with self._lock:
            self._cache[event.id] = event

    def load(self, event_id: str) -> Event | None:
        event_id = self._strip_ext(event_id)
        with self._lock:
            if event_id in self._cache:
                return self._cache[event_id]
        path = self._local_path(event_id)
        if os.path.exists(path):
            with open(path, encoding='utf-8') as f:
                ev = Event.from_dict(json.load(f))
            with self._lock:
                self._cache[ev.id] = ev
            return ev
        if CLOUDINARY_CONFIGURED:
            # Tenta com e sem extensão (compatibilidade com uploads antigos)
            for public_id in (f'{self._CLOUD_PREFIX}{event_id}',
                              f'{self._CLOUD_PREFIX}{event_id}.json'):
                try:
                    res = cloudinary.api.resource(public_id, resource_type='raw')
                    raw = urllib.request.urlopen(res['secure_url']).read()
                    ev  = Event.from_dict(json.loads(raw))
                    with open(self._local_path(ev.id), 'w', encoding='utf-8') as f:
                        f.write(raw.decode())
                    with self._lock:
                        self._cache[ev.id] = ev
                    return ev
                except Exception:
                    continue
        return None

    def list_all(self) -> list[Event]:
        seen: dict[str, Event] = {}  # keyed by ev.id real
        with self._lock:
            seen.update(self._cache)
        if os.path.exists(DATA_DIR):
            for fname in os.listdir(DATA_DIR):
                if not fname.endswith('.json') or fname.startswith('_tmp_'):
                    continue
                eid = self._strip_ext(fname[:-5])  # remove dupla extensão se houver
                if eid not in seen:
                    ev = self.load(eid)
                    if ev and ev.id not in seen:
                        seen[ev.id] = ev
        if CLOUDINARY_CONFIGURED and not seen:
            try:
                result = cloudinary.api.resources(
                    resource_type='raw', type='upload',
                    prefix=self._CLOUD_PREFIX, max_results=200,
                )
                for r in result.get('resources', []):
                    eid = self._strip_ext(r['public_id'].split('/')[-1])
                    if eid not in seen:
                        ev = self.load(eid)
                        if ev and ev.id not in seen:
                            seen[ev.id] = ev
            except Exception as e:
                print(f' * EventStore list: {e}')
        return sorted(seen.values(), key=lambda e: e.created_at, reverse=True)

    def increment_photos(self, event_id: str):
        ev = self.load(event_id)
        if ev:
            ev.photo_count += 1
            self.save(ev)


event_store = EventStore()


# ── Referência de composição (calibrada para mundosenai.png 1080×1920) ────────
_REF_W,      _REF_H      = 1080, 1920
_REF_X,      _REF_Y_BASE =   74,  220
_REF_FOTO_W, _REF_FOTO_H =  900,  550
_REF_BORDA               =   10

# Modo "single": 1 foto grande, recortada ao centro (proporção 900×1100)
_SINGLE_FOTO_H, _SINGLE_Y = 1100, 490
_CAPTURE_MODES = {'strip': 3, 'single': 1}   # modo → nº de fotos

# ── Config por estação (nome do evento, QR overlay URL, modo LGPD) ───────────
_station_config: dict[str, dict] = {}
_cfg_lock = threading.Lock()

_STATION_DEFAULTS = {
    'event_id':        '',
    'event_name':      'SENAI — Totem de Fotos',
    'overlay_qr_url':  '',
    'overlay_qr_label': '',
    'lgpd_mode':       'personal',
}


def get_capture_mode(event_id: str) -> str:
    """Modo de captura vem do evento vinculado; sem evento usa a filipeta."""
    if event_id:
        ev = event_store.load(event_id)
        if ev and ev.capture_mode in _CAPTURE_MODES:
            return ev.capture_mode
    return 'strip'

def get_station_config(station_id: str) -> dict:
    with _cfg_lock:
        if station_id not in _station_config:
            _station_config[station_id] = dict(_STATION_DEFAULTS)
        return dict(_station_config[station_id])

def update_station_config(station_id: str, **kwargs):
    with _cfg_lock:
        cfg = _station_config.setdefault(station_id, dict(_STATION_DEFAULTS))
        cfg.update(kwargs)

# ── Fundos por evento ─────────────────────────────────────────────────────────
_event_fundo_cache: dict[str, str] = {}
_efundo_lock = threading.Lock()


def _event_fundo_cache_path(event_id: str) -> str:
    return os.path.join(FUNDO_CACHE_DIR, f'event_{event_id[:8]}.png')


def get_station_fundo(station_id: str) -> str:
    """Retorna o fundo do evento vinculado à estação, ou o padrão."""
    cfg      = get_station_config(station_id)
    event_id = cfg.get('event_id', '')
    if event_id:
        with _efundo_lock:
            if event_id in _event_fundo_cache:
                return _event_fundo_cache[event_id]

        ev_cache = _event_fundo_cache_path(event_id)
        if os.path.exists(ev_cache):
            with _efundo_lock:
                _event_fundo_cache[event_id] = ev_cache
            return ev_cache

        if CLOUDINARY_CONFIGURED:
            ev = event_store.load(event_id)
            if ev and ev.background_id:
                try:
                    resource = cloudinary.api.resource(ev.background_id)
                    urllib.request.urlretrieve(resource['secure_url'], ev_cache)
                    with _efundo_lock:
                        _event_fundo_cache[event_id] = ev_cache
                    return ev_cache
                except Exception:
                    pass

    # Sem fundo próprio: foto única usa o padrão Expo SENAI; filipeta usa o 9:16
    return FUNDO_DEFAULT_SINGLE if get_capture_mode(event_id) == 'single' else FUNDO_DEFAULT


def get_bg_size(path: str) -> tuple[int, int]:
    """(largura, altura) do fundo, lendo só o cabeçalho da imagem."""
    for p in (path, FUNDO_DEFAULT):
        try:
            with Image.open(p) as im:
                return im.size
        except Exception:
            continue
    return _REF_W, _REF_H


def single_slot(bg_w: int, bg_h: int) -> tuple[int, int, int, int]:
    """Área (x, y, w, h) da foto única, conforme o formato do fundo."""
    if bg_h / bg_w >= 1.6:   # 9:16 (Stories): posição calibrada para mundosenai.png
        sx, sy = bg_w / _REF_W, bg_h / _REF_H
        return int(_REF_X * sx), int(_SINGLE_Y * sy), int(_REF_FOTO_W * sx), int(_SINGLE_FOTO_H * sy)
    # 4:5 (feed) ou outro formato mais largo: margens proporcionais, com espaço p/ topo e QR
    x = int(bg_w * 0.055)
    y = int(bg_h * 0.11)
    return x, y, bg_w - 2 * x, int(bg_h * 0.76)


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
                'codigo':         None,
            }
        return _sessions[sid]


# ── Registro de fotos (photo_id → metadados) ──────────────────────────────────
_photo_registry: dict[str, dict] = {}
_reg_lock = threading.Lock()

def registrar_foto(photo_id: str, cloudinary_url: str | None,
                   station_id: str, event_id: str, event_name: str):
    with _reg_lock:
        _photo_registry[photo_id] = {
            'cloudinary_url': cloudinary_url,
            'station_id':     station_id,
            'event_id':       event_id,
            'event_name':     event_name,
        }

def get_foto_info(photo_id: str) -> dict | None:
    with _reg_lock:
        return _photo_registry.get(photo_id)

def get_foto_cloudinary_url(photo_id: str) -> str | None:
    if not CLOUDINARY_CONFIGURED:
        return None
    info = get_foto_info(photo_id)
    if info and info.get('cloudinary_url'):
        return info['cloudinary_url']
    # Tenta com event_id (novo path) e depois legado
    event_id = (info or {}).get('event_id', '')
    candidates = []
    if event_id:
        candidates.append(f'cabinefotos/events/{event_id}/fotos/{photo_id}')
    candidates.append(f'cabinefotos/fotos/{photo_id}')
    for pid in candidates:
        try:
            return cloudinary.api.resource(pid)['secure_url']
        except Exception:
            pass
    return None


# ── Cloudinary upload de fotos ────────────────────────────────────────────────
def upload_foto_cloudinary(filepath: str, photo_id: str,
                           event_id: str = '', event_name: str = '',
                           codigo: str = '') -> str | None:
    if not CLOUDINARY_CONFIGURED:
        return None
    public_id = (f'cabinefotos/events/{event_id}/fotos/{photo_id}'
                 if event_id else f'cabinefotos/fotos/{photo_id}')
    try:
        result = cloudinary.uploader.upload(
            filepath,
            public_id=public_id,
            overwrite=False,
            resource_type='image',
            context=f'event_name={event_name}',
            tags=[f'code_{codigo}'] if codigo else None,
        )
        return result['secure_url']
    except Exception as e:
        print(f" * Cloudinary foto upload error: {e}")
        return None


# ── Código curto por foto (entrega sem celular) ───────────────────────────────
# Sem 0/O/1/I/L para não confundir ao anotar à mão.
_CODE_ALPHABET = 'ABCDEFGHJKMNPQRSTUVWXYZ23456789'
_CODE_LEN      = 6
CODES_PATH     = os.path.join(app.root_path, 'data', 'codes.json')
_codes: dict[str, dict] = {}        # código → {photo_id, event_id}
_codes_lock    = threading.Lock()
_code_attempts: dict[str, list[float]] = {}   # ip → timestamps de erros

try:
    with open(CODES_PATH, encoding='utf-8') as _f:
        _codes.update(json.load(_f))
except (OSError, ValueError):
    pass


def _save_codes_locked():
    try:
        tmp = CODES_PATH + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as f:
            json.dump(_codes, f)
        os.replace(tmp, CODES_PATH)
    except OSError as e:
        print(f' * codes.json: {e}')


def gerar_codigo(photo_id: str, event_id: str) -> str:
    with _codes_lock:
        while True:
            codigo = ''.join(secrets.choice(_CODE_ALPHABET) for _ in range(_CODE_LEN))
            if codigo not in _codes:
                break
        _codes[codigo] = {'photo_id': photo_id, 'event_id': event_id}
        _save_codes_locked()
    return codigo


def _event_name_of(event_id: str) -> str:
    ev = event_store.load(event_id) if event_id else None
    return ev.name if ev else 'SENAI — Totem de Fotos'


def resolver_codigo(codigo: str) -> str | None:
    """Devolve o photo_id do código (e garante o registro em memória), ou None."""
    codigo = ''.join(c for c in codigo.upper() if c.isalnum())
    if len(codigo) != _CODE_LEN:
        return None

    with _codes_lock:
        entry = _codes.get(codigo)

    cloud_url = None
    if not entry and CLOUDINARY_CONFIGURED:
        # Fallback após restart (codes.json perdido): a tag no Cloudinary guarda o código
        try:
            res = cloudinary.api.resources_by_tag(f'code_{codigo}', max_results=1)
            r = (res.get('resources') or [None])[0]
            if r:
                parts     = r['public_id'].split('/')
                cloud_url = r['secure_url']
                entry = {'photo_id': parts[-1],
                         'event_id': parts[2] if len(parts) >= 5 and parts[1] == 'events' else ''}
                with _codes_lock:
                    _codes[codigo] = entry
                    _save_codes_locked()
        except Exception as e:
            print(f' * resolver_codigo: {e}')

    if not entry:
        return None
    if not get_foto_info(entry['photo_id']):
        registrar_foto(entry['photo_id'], cloud_url, '', entry['event_id'],
                       _event_name_of(entry['event_id']))
    return entry['photo_id']


def _client_ip() -> str:
    fwd = request.headers.get('X-Forwarded-For', '')
    return (fwd.split(',')[0].strip() if fwd else request.remote_addr) or '?'


def _code_rate_limited(ip: str) -> bool:
    now = time.time()
    with _codes_lock:
        hist = [t for t in _code_attempts.get(ip, []) if now - t < 600]
        _code_attempts[ip] = hist
        return len(hist) >= 10


def _code_register_fail(ip: str):
    with _codes_lock:
        _code_attempts.setdefault(ip, []).append(time.time())


# ── Telão (slideshow das fotos do evento) ─────────────────────────────────────
_TELAO_MAX    = 80
_telao_fotos: dict[str, list[dict]] = {}   # event_id → fotos em ordem cronológica
_telao_seeded: set[str] = set()
_telao_lock   = threading.Lock()


def _telao_add(event_id: str, photo_id: str, url: str):
    with _telao_lock:
        lst = _telao_fotos.setdefault(event_id, [])
        if any(p['id'] == photo_id for p in lst):
            return
        lst.append({'id': photo_id, 'url': url})
        del lst[:-_TELAO_MAX]


def _telao_cloud_url(url: str) -> str:
    return url.replace('/upload/', '/upload/c_limit,w_1400,q_auto/', 1)


def _telao_seed(event_id: str):
    """Carrega as últimas fotos do evento uma vez por processo (após restart)."""
    with _telao_lock:
        if event_id in _telao_seeded:
            return
        _telao_seeded.add(event_id)
    found: list[tuple[str, str]] = []
    if CLOUDINARY_CONFIGURED:
        prefix = f'cabinefotos/events/{event_id}/fotos/' if event_id else 'cabinefotos/fotos/'
        try:
            res = cloudinary.api.resources(type='upload', prefix=prefix,
                                           max_results=_TELAO_MAX, direction='desc')
            for r in reversed(res.get('resources', [])):
                found.append((r['public_id'].split('/')[-1], _telao_cloud_url(r['secure_url'])))
        except Exception as e:
            print(f' * telao seed: {e}')
    else:
        base = os.path.join(PHOTOS_DIR, event_id) if event_id else PHOTOS_DIR
        if os.path.isdir(base):
            fnames = sorted((f for f in os.listdir(base)
                             if f.startswith('foto_') and f.endswith('.png')),
                            key=lambda f: os.path.getmtime(os.path.join(base, f)))
            sub = f'{event_id}/' if event_id else ''
            for f in fnames[-_TELAO_MAX:]:
                found.append((f[:-4], f'/static/photos/{sub}{f}'))
    for pid, url in found:
        _telao_add(event_id, pid, url)


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


# ── Admin auth ────────────────────────────────────────────────────────────────
def require_admin(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        # Se ADMIN_PASSWORD não está definido, libera sem login (setup inicial)
        if ADMIN_PASSWORD and not session.get('admin'):
            return redirect(f'/admin/login?next={request.path}')
        return f(*args, **kwargs)
    return wrapper


# ── Rotas ─────────────────────────────────────────────────────────────────────

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/configuracao', methods=['GET', 'POST'])
@require_admin
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
        eventos=[e for e in event_store.list_all() if e.active],
        msg=msg,
    )


@app.route('/station_info')
def station_info():
    station_id = request.args.get('station', 'default')
    cfg  = get_station_config(station_id)
    modo = get_capture_mode(cfg.get('event_id', ''))
    crop_aspect = None
    if modo == 'single':
        _, _, sw, sh = single_slot(*get_bg_size(get_station_fundo(station_id)))
        crop_aspect = sw / sh
    cfg.update(capture_mode=modo, photos_total=_CAPTURE_MODES[modo], crop_aspect=crop_aspect)
    return jsonify(cfg)


@app.route('/config_estacao', methods=['POST'])
def config_estacao():
    station_id = request.form.get('station_id', 'default')
    event_id = request.form.get('event_id', '').strip()
    update_station_config(
        station_id,
        event_id=event_id,
        event_name=request.form.get('event_name', '').strip() or 'SENAI — Totem de Fotos',
        overlay_qr_url=request.form.get('overlay_qr_url', '').strip(),
        overlay_qr_label=request.form.get('overlay_qr_label', '').strip(),
        lgpd_mode=request.form.get('lgpd_mode', 'personal'),
    )
    return jsonify(success=True)




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

    cfg      = get_station_config(station_id)
    modo     = get_capture_mode(cfg.get('event_id', ''))
    total    = _CAPTURE_MODES[modo]
    single   = (modo == 'single')

    if single:
        # Recorta o centro do frame na proporção do slot do fundo e descarta sequência antiga
        sess['foto_seq'] = 0
        _, _, slot_w, slot_h = single_slot(*get_bg_size(get_station_fundo(station_id)))
        fh, fw = frame.shape[:2]
        alvo   = slot_w / slot_h
        if fw / fh > alvo:
            novo_w = int(fh * alvo)
            x0     = (fw - novo_w) // 2
            frame  = frame[:, x0 : x0 + novo_w]
        else:
            novo_h = int(fw / alvo)
            y0     = (fh - novo_h) // 2
            frame  = frame[y0 : y0 + novo_h, :]
    else:
        slot_w, slot_h = _REF_FOTO_W, _REF_FOTO_H

    # Salva cada foto no tamanho do slot; o composite redimensiona para o fundo atual
    frame    = cv2.resize(frame, (slot_w, slot_h))
    sid_pfx  = sid[:8]
    seq_name = f'{sid_pfx}_foto_{sess["foto_seq"]}.png'
    cv2.imwrite(os.path.join(PHOTOS_DIR, seq_name), frame)
    sess['nome_foto'] = seq_name
    sess['foto_seq'] += 1

    if sess['foto_seq'] >= total:
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
        if single:
            x, y_base, w, h = single_slot(bg_w, bg_h)
        else:
            x      = int(_REF_X      * sx)
            y_base = int(_REF_Y_BASE * sy)
            w      = int(_REF_FOTO_W * sx)
            h      = int(_REF_FOTO_H * sy)
        borda  = max(4, int(_REF_BORDA * sy)) // 2 * 2   # par, para o recorte da moldura encaixar

        nome_comp = f'foto_{station_id[:8]}_{uuid.uuid4()}.png'
        photo_id  = nome_comp[:-4]
        bg_foto   = np.zeros((h + borda, w + borda, 3), dtype=np.uint8)

        y = y_base
        for seq in range(total):
            foto = cv2.imread(os.path.join(PHOTOS_DIR, f'{sid_pfx}_foto_{seq}.png'))
            if foto is None:
                continue
            foto = cv2.resize(foto, (w, h))
            moldura[y - borda//2 : y + h + borda//2,
                    x - borda//2 : x + w + borda//2] = bg_foto
            moldura[y : y + h, x : x + w] = foto
            y += h + max(10, int(15 * sy))

        # QR overlay no canto inferior direito
        event_id      = cfg.get('event_id', '')
        overlay_url   = cfg.get('overlay_qr_url', '')
        overlay_label = cfg.get('overlay_qr_label', '')
        if overlay_url:
            moldura = aplicar_qr_overlay(moldura, overlay_url, overlay_label, bg_w, bg_h)

        # Salva composite em subpasta do evento (ou raiz se sem evento)
        if event_id:
            event_dir = os.path.join(PHOTOS_DIR, event_id)
            os.makedirs(event_dir, exist_ok=True)
            comp_path   = os.path.join(event_dir, nome_comp)
            nome_local  = os.path.join(event_id, nome_comp)
        else:
            comp_path  = os.path.join(PHOTOS_DIR, nome_comp)
            nome_local = nome_comp
        cv2.imwrite(comp_path, moldura)

        event_name = cfg.get('event_name', 'SENAI — Totem de Fotos')
        cloud_url  = None
        codigo     = gerar_codigo(photo_id, event_id)
        try:
            cloud_url = upload_foto_cloudinary(comp_path, photo_id, event_id, event_name, codigo)
            registrar_foto(photo_id, cloud_url, station_id, event_id, event_name)
            _telao_add(event_id, photo_id,
                       _telao_cloud_url(cloud_url) if cloud_url
                       else '/static/photos/' + nome_local.replace(os.sep, '/'))
            if event_id:
                threading.Thread(
                    target=event_store.increment_photos,
                    args=(event_id,),
                    daemon=True,
                ).start()
        except Exception as e:
            print(f' * upload_foto composite error: {e}')

        sess['nome_foto']      = nome_local
        sess['photo_id']       = photo_id
        sess['cloudinary_url'] = cloud_url
        sess['codigo']         = codigo
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
    return jsonify(foto_capturada=fotocap, codigo=sess.get('codigo') if fotocap else None)


@app.route('/f', methods=['GET', 'POST'])
@app.route('/f/<codigo>')
def buscar_codigo(codigo=None):
    erro = None
    if request.method == 'POST':
        codigo = request.form.get('codigo', '')
    if codigo is not None:
        ip = _client_ip()
        if _code_rate_limited(ip):
            erro = 'Muitas tentativas. Aguarde alguns minutos e tente de novo.'
        else:
            photo_id = resolver_codigo(codigo)
            if photo_id:
                return redirect(f'/foto/{photo_id}')
            _code_register_fail(ip)
            erro = 'Código não encontrado. Confira as letras e números.'
    return render_template('codigo.html', erro=erro)


@app.route('/telao')
def telao():
    return render_template('telao.html')


@app.route('/telao/fotos')
def telao_fotos():
    event_id = request.args.get('event')
    if event_id is None:
        event_id = get_station_config(request.args.get('station', 'default')).get('event_id', '')
    event_id = event_id.strip()
    _telao_seed(event_id)
    with _telao_lock:
        fotos = list(_telao_fotos.get(event_id, []))
    return jsonify(event_id=event_id,
                   nome=_event_name_of(event_id) if event_id else '',
                   fotos=fotos)


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


# ── Admin rotas ───────────────────────────────────────────────────────────────

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    next_url = request.args.get('next', '/configuracao')
    error    = None
    if request.method == 'POST':
        next_url = request.form.get('next', '/configuracao')
        if not ADMIN_PASSWORD:
            error = 'Senha de admin não configurada (defina a variável ADMIN_PASSWORD).'
        elif request.form.get('password') == ADMIN_PASSWORD:
            session['admin'] = True
            return redirect(next_url)
        else:
            error = 'Senha incorreta.'
    return render_template('admin_login.html', error=error, next_url=next_url)


@app.route('/admin/logout')
def admin_logout():
    session.pop('admin', None)
    return redirect('/index')


@app.route('/admin')
@require_admin
def admin_hub():
    return redirect('/admin/eventos')


@app.route('/admin/eventos')
@require_admin
def admin_eventos():
    eventos = event_store.list_all()
    return render_template('admin_eventos.html',
                           eventos=eventos,
                           cloudinary_ok=CLOUDINARY_CONFIGURED)


@app.route('/admin/eventos/novo', methods=['POST'])
@require_admin
def admin_evento_novo():
    name = request.form.get('name', '').strip()
    if not name:
        return redirect('/admin/eventos')

    ev = Event(
        id=uuid.uuid4().hex[:12],
        name=name,
        created_at=datetime.datetime.now().isoformat(timespec='seconds'),
        qr_url=request.form.get('qr_url', '').strip(),
        qr_label=request.form.get('qr_label', '').strip(),
        lgpd_mode=request.form.get('lgpd_mode', 'personal'),
        capture_mode=(request.form.get('capture_mode')
                      if request.form.get('capture_mode') in _CAPTURE_MODES else 'strip'),
    )

    # Upload de fundo do evento (opcional)
    if 'fundo' in request.files and request.files['fundo'].filename:
        f   = request.files['fundo']
        ext = f.filename.rsplit('.', 1)[-1].lower()
        if ext in ALLOWED_EXT:
            tmp = os.path.join(FUNDO_CACHE_DIR, f'_ev_tmp_{ev.id}.{ext}')
            f.save(tmp)
            if cv2.imread(tmp) is not None:
                dest = _event_fundo_cache_path(ev.id)
                shutil.move(tmp, dest)
                if CLOUDINARY_CONFIGURED:
                    try:
                        public_id = f'cabinefotos/events/{ev.id}/fundo'
                        cloudinary.uploader.upload(dest, public_id=public_id,
                                                   overwrite=True, resource_type='image')
                        ev.background_id = public_id
                    except Exception as e:
                        print(f' * Event fundo upload: {e}')
                else:
                    ev.background_id = dest
            elif os.path.exists(tmp):
                os.unlink(tmp)

    event_store.save(ev)
    return redirect(f'/admin/eventos/{ev.id}')


@app.route('/admin/eventos/<event_id>')
@require_admin
def admin_evento_fotos(event_id):
    ev = event_store.load(event_id)
    if not ev:
        return redirect('/admin/eventos')

    photos = []
    if CLOUDINARY_CONFIGURED:
        try:
            prefix = f'cabinefotos/events/{event_id}/fotos/'
            result = cloudinary.api.resources(type='upload', prefix=prefix,
                                              max_results=500, direction='asc')
            for r in result.get('resources', []):
                url   = r['secure_url']
                thumb = url.replace('/upload/', '/upload/w_320,c_limit/', 1)
                photos.append({'url': url, 'thumb': thumb,
                               'name': r['public_id'].split('/')[-1]})
        except Exception as e:
            print(f" * Admin event photos: {e}")
    else:
        event_dir = os.path.join(PHOTOS_DIR, event_id)
        if os.path.exists(event_dir):
            fnames = sorted(
                (f for f in os.listdir(event_dir) if f.startswith('foto_') and f.endswith('.png')),
                key=lambda f: os.path.getmtime(os.path.join(event_dir, f)),
                reverse=False,
            )
            for fname in fnames:
                photos.append({'url': f'/static/photos/{event_id}/{fname}',
                               'thumb': f'/static/photos/{event_id}/{fname}',
                               'name': fname})

    return render_template('admin_galeria.html',
                           evento=ev,
                           photos=photos,
                           total=len(photos),
                           cloudinary_ok=CLOUDINARY_CONFIGURED)


@app.route('/admin/eventos/<event_id>/modo', methods=['POST'])
@require_admin
def admin_evento_modo(event_id):
    ev   = event_store.load(event_id)
    modo = request.form.get('capture_mode', '')
    if ev and modo in _CAPTURE_MODES:
        ev.capture_mode = modo
        event_store.save(ev)
    return redirect('/admin/eventos')


@app.route('/admin/eventos/<event_id>/arquivar', methods=['POST'])
@require_admin
def admin_evento_arquivar(event_id):
    ev = event_store.load(event_id)
    if ev:
        ev.active = False
        event_store.save(ev)
    return redirect('/admin/eventos')


@app.route('/admin/eventos/<event_id>/reativar', methods=['POST'])
@require_admin
def admin_evento_reativar(event_id):
    ev = event_store.load(event_id)
    if ev:
        ev.active = True
        event_store.save(ev)
    return redirect('/admin/eventos')


@app.route('/admin/fotos')
@require_admin
def admin_fotos_sem_evento():
    """Galeria de fotos sem evento vinculado (path legado cabinefotos/fotos/)."""
    photos = []
    if CLOUDINARY_CONFIGURED:
        try:
            result = cloudinary.api.resources(
                type='upload', prefix='cabinefotos/fotos/',
                max_results=500, direction='asc',
            )
            for r in result.get('resources', []):
                url   = r['secure_url']
                thumb = url.replace('/upload/', '/upload/w_320,c_limit/', 1)
                photos.append({'url': url, 'thumb': thumb,
                               'name': r['public_id'].split('/')[-1]})
        except Exception as e:
            print(f" * Admin fotos sem evento: {e}")
    else:
        fnames = sorted(
            (f for f in os.listdir(PHOTOS_DIR)
             if f.startswith('foto_') and f.endswith('.png')
             and os.path.isfile(os.path.join(PHOTOS_DIR, f))),
            key=lambda f: os.path.getmtime(os.path.join(PHOTOS_DIR, f)),
            reverse=False,
        )
        for fname in fnames:
            photos.append({'url': f'/static/photos/{fname}',
                           'thumb': f'/static/photos/{fname}',
                           'name': fname})
    return render_template('admin_galeria.html',
                           evento=None,
                           photos=photos,
                           total=len(photos),
                           cloudinary_ok=CLOUDINARY_CONFIGURED)


@app.route('/admin/fotos/zip')
@require_admin
def admin_fotos_sem_evento_zip():
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if CLOUDINARY_CONFIGURED:
            try:
                result = cloudinary.api.resources(
                    type='upload', prefix='cabinefotos/fotos/', max_results=500)
                for r in result.get('resources', []):
                    fname    = r['public_id'].split('/')[-1] + '.png'
                    img_data = urllib.request.urlopen(r['secure_url']).read()
                    zf.writestr(fname, img_data)
            except Exception as e:
                print(f" * Fotos sem evento ZIP: {e}")
        else:
            for fname in os.listdir(PHOTOS_DIR):
                if fname.startswith('foto_') and fname.endswith('.png') \
                        and os.path.isfile(os.path.join(PHOTOS_DIR, fname)):
                    zf.write(os.path.join(PHOTOS_DIR, fname), fname)
    buf.seek(0)
    return send_file(buf, mimetype='application/zip',
                     as_attachment=True, download_name='fotos_sem_evento.zip')


@app.route('/admin/eventos/<event_id>/zip')
@require_admin
def admin_evento_zip(event_id):
    ev  = event_store.load(event_id)
    buf = BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        if CLOUDINARY_CONFIGURED:
            try:
                prefix = f'cabinefotos/events/{event_id}/fotos/'
                result = cloudinary.api.resources(type='upload', prefix=prefix, max_results=500)
                for r in result.get('resources', []):
                    fname    = r['public_id'].split('/')[-1] + '.png'
                    img_data = urllib.request.urlopen(r['secure_url']).read()
                    zf.writestr(fname, img_data)
            except Exception as e:
                print(f" * Event ZIP: {e}")
        else:
            event_dir = os.path.join(PHOTOS_DIR, event_id)
            if os.path.exists(event_dir):
                for fname in os.listdir(event_dir):
                    if fname.endswith('.png'):
                        zf.write(os.path.join(event_dir, fname), fname)
    buf.seek(0)
    zip_name = f'{ev.name.replace(" ", "_")}.zip' if ev else 'fotos.zip'
    return send_file(buf, mimetype='application/zip',
                     as_attachment=True, download_name=zip_name)


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 2205))
    app.run(host='0.0.0.0', port=port, threaded=True)
