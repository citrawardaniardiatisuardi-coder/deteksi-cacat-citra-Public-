import os
import math
import urllib.parse
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Pastikan folder kerja = folder file ini (aman di Colab maupun Streamlit Cloud)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import inti

# ---------------- IDENTITAS ----------------
NAMA = 'Citra Wardani Ardiati'
NIM = '2125640032'

# ---------------- PALET WARNA (bisa dipilih di sidebar) ----------------
PALET = {
    'Choco Maroon':      {'utama': '#7A0E1E', 'tua': '#3B2116', 'krem': '#FFF6E6', 'krem2': '#F6E3C3',
                          'aksen': '#D9A441', 'lembut': '#E9C9A6'},
    'Strawberry Choco':  {'utama': '#8E1B3A', 'tua': '#3A1F1A', 'krem': '#FFF4F1', 'krem2': '#F9DCDC',
                          'aksen': '#E8899E', 'lembut': '#F3C1CB'},
    'Caramel Latte':     {'utama': '#7B2D26', 'tua': '#4A2E1F', 'krem': '#FFF8EE', 'krem2': '#EFDCC0',
                          'aksen': '#C98A4B', 'lembut': '#E2C29C'},
    'Pink Denim':        {'utama': '#A8475A', 'tua': '#22324A', 'krem': '#FBF5F0', 'krem2': '#F3DADB',
                          'aksen': '#E8A1AE', 'lembut': '#C9D6E8'},
}

# ---------------- MOTIF BINTANG KAIN (pink & jeans) ----------------
PINK = ('#B85A64', '#7E3540', '#E59AA3')       # (warna kain, tepi gelap, benang terang)
JEANS = ('#4C6E9C', '#24395A', '#9DB6D6')


def kain_bintang(warna, gelap, terang, ukuran=120, putar=0, pid='a', kelas='bintang'):
    """Tambalan bintang dari kain denim: anyaman twill miring + jahitan putih putus-putus."""
    def titik(r_luar, r_dalam):                  # 10 titik sudut bintang (koordinat kutub)
        p = []
        for i in range(10):
            r = r_luar if i % 2 == 0 else r_dalam
            a = -math.pi / 2 + i * math.pi / 5
            p.append(f'{50 + r * math.cos(a):.2f},{53 + r * math.sin(a):.2f}')
        return ' '.join(p)
    return (f'<svg class="{kelas}" viewBox="0 0 100 100" width="{ukuran}" height="{ukuran}" '
            f'style="transform:rotate({putar}deg)" aria-hidden="true" xmlns="http://www.w3.org/2000/svg">'
            f'<defs><pattern id="tw{pid}" width="3.2" height="3.2" patternUnits="userSpaceOnUse" patternTransform="rotate(-38)">'
            f'<rect width="3.2" height="3.2" fill="{warna}"/><rect width="3.2" height="1.1" fill="{terang}" opacity=".55"/>'
            f'<rect y="2.2" width="3.2" height=".5" fill="{gelap}" opacity=".35"/></pattern>'
            f'<filter id="fr{pid}" x="-10%" y="-10%" width="120%" height="120%">'
            f'<feTurbulence type="fractalNoise" baseFrequency="1.6" numOctaves="2" seed="3"/>'
            f'<feDisplacementMap in="SourceGraphic" scale="1.8"/></filter>'
            f'<filter id="bay{pid}"><feDropShadow dx="1.5" dy="2.5" stdDeviation="1.6" flood-opacity=".28"/></filter></defs>'
            f'<g filter="url(#bay{pid})"><polygon points="{titik(47, 23.5)}" fill="url(#tw{pid})" stroke="{gelap}" '
            f'stroke-width="1.2" stroke-linejoin="round" filter="url(#fr{pid})"/></g>'
            f'<polygon points="{titik(39.5, 19.8)}" fill="none" stroke="#FFF8F2" stroke-width="1.5" '
            f'stroke-dasharray="3.2 2.2" stroke-linejoin="round" stroke-linecap="round" opacity=".92"/></svg>')


def pola_latar():
    """Ubin latar: dua bintang kecil (pink & jeans) samar, diulang ke seluruh halaman."""
    b1 = kain_bintang(*PINK, 30, -12, 'l1').replace('<svg ', '<svg x="18" y="20" ', 1)
    b2 = kain_bintang(*JEANS, 24, 16, 'l2').replace('<svg ', '<svg x="128" y="122" ', 1)
    ubin = (f'<svg xmlns="http://www.w3.org/2000/svg" width="220" height="220"><g opacity=".38">{b1}{b2}</g></svg>')
    return 'url("data:image/svg+xml,' + urllib.parse.quote(ubin) + '")'


st.set_page_config(page_title=f'Deteksi Cacat · {NAMA}', page_icon='🔎', layout='wide')

with st.sidebar:
    st.markdown(f"""<div class='logo'>
    <svg viewBox="0 0 60 60" aria-hidden="true"><circle cx="30" cy="30" r="27" fill="none" stroke="#D9A441" stroke-width="3"/>
    <circle cx="30" cy="30" r="19" fill="#7A0E1E"/>
    <g stroke="#FFF6E6" stroke-width="1.6" opacity=".55">
    <line x1="30" y1="11" x2="38" y2="27"/><line x1="46.5" y1="20.5" x2="36" y2="34"/><line x1="46.5" y1="39.5" x2="30" y2="38"/>
    <line x1="30" y1="49" x2="22" y2="33"/><line x1="13.5" y1="39.5" x2="24" y2="26"/><line x1="13.5" y1="20.5" x2="30" y2="22"/></g>
    <text x="30" y="34" text-anchor="middle" font-size="11" font-weight="800" fill="#FFF6E6" font-family="sans-serif">CWA</text></svg>
    <div class='teks'><b>Lensa Inspeksi</b><span>Matematika 3</span></div></div>
    <div class='identitas'>{NAMA}<small>NIM {NIM}</small></div>""", unsafe_allow_html=True)
    nama_palet = st.selectbox('🎨 Tema warna', list(PALET.keys()))
W = PALET[nama_palet]

# ---------------- GAYA (CSS) dari file gaya.css + warna tema ----------------
FONT = ("https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght,SOFT,WONK@0,9..144,100..900,0..100,0..1;1,9..144,100..900,0..100,0..1"
        "&family=Plus+Jakarta+Sans:wght@400;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap")
warna = '; '.join(f'--{k}: {v}' for k, v in W.items())
with open('gaya.css') as f:
    css = f.read()
st.markdown(f"<style>@import url('{FONT}'); :root {{ {warna}; --pola-bintang: {pola_latar()}; }}\n{css}</style>", unsafe_allow_html=True)


ROOT = os.getcwd()
DAFTAR_KELAS = sorted(os.listdir('kelas'), key=lambda k: int(k.replace('Class', '')))


@st.cache_resource
def muat_model(kelas):
    folder = os.path.join(ROOT, 'kelas', kelas, 'model')
    model = {}
    for f in sorted(os.listdir(folder)):
        if f.startswith('rf_') and f.endswith('.joblib'):
            paket = joblib.load(os.path.join(folder, f))
            model[int(paket['resolusi'])] = paket
    return dict(sorted(model.items(), reverse=True))


@st.cache_data
def muat_ringkasan(kelas):
    p = os.path.join(ROOT, 'kelas', kelas, 'hasil', 'ringkasan.csv')
    return pd.read_csv(p) if os.path.exists(p) else None


# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("<div class='judul-sb'>Pengaturan kamera</div>", unsafe_allow_html=True)
    KELAS = st.selectbox('🧵 Kelas tekstur', DAFTAR_KELAS)

# Pindah ke folder kelas terpilih: semua halaman membaca model/, hasil/, contoh/ milik kelas ini
os.chdir(os.path.join(ROOT, 'kelas', KELAS))
MODEL = muat_model(KELAS)
RINGKAS = muat_ringkasan(KELAS)

with st.sidebar:
    resolusi = st.selectbox('Resolusi kamera', list(MODEL.keys()), format_func=lambda n: f'{n} × {n} piksel')
    paket = MODEL[resolusi]
    tau = st.slider('Ambang keputusan τ', 0.05, 0.95, float(paket['tau']), 0.05, key=f'tau_{KELAS}_{resolusi}')
    st.markdown(f"""<div class='judul-sb'>Tentang model</div>
    <div class='info-sb'>Dataset: <b>DAGM 2007 · {KELAS}</b><br>Grid petak: <b>8 × 8</b><br>
    Fitur: <b>32</b> (GLCM + statistik)<br>Model: <b>Random Forest</b> 200 pohon</div>
    <div class='judul-sb'>Catatan</div>
    <div class='info-sb'>Prototipe indikatif,<br>bukan sertifikasi alat.</div>""", unsafe_allow_html=True)

# ---------------- LENSA PIKSEL (gambar kecil yang ikut resolusi) ----------------
def campur_hex(a, b, t):
    """Warna di antara a dan b: (1 - t)·a + t·b, per kanal RGB."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round((1 - t) * x + t * y):02x}' for x, y in zip(ca, cb))


def lensa_svg(n, W):
    """Tekstur 64×64 berisi satu bercak cacat, dirata-rata per blok seperti resize (512→16, 64→2 blok)."""
    k = max(2, n // 32)
    yy, xx = np.mgrid[0:64, 0:64]
    tekstur = 0.45 + 0.4 * np.sin(2 * np.pi * (xx + yy) / 10)            # garis tekstur
    cacat = np.exp(-((xx - 40) ** 2 + (yy - 22) ** 2) / 90.0)               # bercak halus
    f = tekstur * (1 - cacat) + 0.72 * cacat
    s_ = 64 // k
    g = f.reshape(k, s_, k, s_).mean(axis=(1, 3))
    u = 200 / k
    kotak = ''.join(f'<rect x="{c * u:.2f}" y="{r * u:.2f}" width="{u + .4:.2f}" height="{u + .4:.2f}" '
                    f'fill="{campur_hex(W["krem"], W["tua"], float(np.clip(g[r, c], 0, 1)))}"/>'
                    for r in range(k) for c in range(k))
    return (f'<svg viewBox="-12 -12 224 224" role="img" aria-label="Pratinjau lensa {n} piksel">'
            f'<defs><clipPath id="kaca"><circle cx="100" cy="100" r="96"/></clipPath></defs>'
            f'<circle cx="100" cy="100" r="108" fill="{W["utama"]}"/>'
            f'<g clip-path="url(#kaca)">{kotak}</g>'
            f'<circle cx="100" cy="100" r="96" fill="none" stroke="{W["aksen"]}" stroke-width="4"/>'
            f'<circle cx="100" cy="100" r="108" fill="none" stroke="{W["aksen"]}" stroke-width="2" stroke-dasharray="2 6"/></svg>')


# ---------------- HERO ----------------
st.markdown(f"""
<div class='hero'>
  <span class='sudut tl'></span><span class='sudut tr'></span><span class='sudut bl'></span><span class='sudut br'></span>
  <div class='isi'>
    <div class='layar-atas'><span class='rec'>REC</span><span>{resolusi}×{resolusi} px</span><span>τ {tau:.2f}</span><span>DAGM 2007 · {KELAS}</span></div>
    <div class='judul'>Pengaruh <i>Resolusi Kamera</i> pada Deteksi Cacat</div>
    <div class='sub'>Seberapa buram kamera boleh dibuat sebelum cacat tak lagi terlihat?</div>
    <div class='pemilik'>{NAMA}<span>{NIM}</span> · Tugas Matematika 3</div>
    <div class='alur'>
      <span><b>1</b>Resize</span><span><b>2</b>Normalisasi</span><span><b>3</b>Segmentasi Otsu</span>
      <span><b>4</b>GLCM</span><span><b>5</b>Random Forest</span>
    </div>
  </div>
  <div class='tempelan' aria-hidden='true'>
    {kain_bintang(*PINK, 74, -10, 'h1', 'bintang b1')}{kain_bintang(*JEANS, 52, 14, 'h2', 'bintang b2')}{kain_bintang(*PINK, 34, 22, 'h3', 'bintang b3')}
  </div>
  <div class='lensa'>{lensa_svg(resolusi, W)}<small>rata-rata blok s = {512 // resolusi}</small></div>
</div>""", unsafe_allow_html=True)

# ---------------- KARTU ANGKA ----------------
r = None
if RINGKAS is not None and (RINGKAS['resolusi'] == resolusi).any():
    r = RINGKAS[RINGKAS['resolusi'] == resolusi].iloc[0]
f1 = f"{r['f1']:.3f}" if r is not None else '–'
auc = f"{r['auc']:.3f}" if r is not None else '–'
st.markdown(f"""
<div class='kpi-baris'>
  <div class='kpi'><div class='lbl'>Resolusi aktif</div><div class='nil'>{resolusi}²</div>
    <div class='ket'>{resolusi * resolusi:,} piksel</div></div>
  <div class='kpi'><div class='lbl'>Ambang τ</div><div class='nil'>{tau:.2f}</div>
    <div class='ket'>τ* hasil validasi = {paket['tau']:.2f}</div></div>
  <div class='kpi'><div class='lbl'>Skor F1 uji</div><div class='nil'>{f1}</div>
    <div class='ket'>gabungan presisi & recall</div></div>
  <div class='kpi'><div class='lbl'>AUC uji</div><div class='nil'>{auc}</div>
    <div class='ket'>0,5 = tebakan koin</div></div>
</div>""", unsafe_allow_html=True)

# ---------------- TAB ----------------
tab_deteksi, tab_eval, tab_trade, tab_hist, tab_rumus = st.tabs(
    ['🔍 Deteksi', '📊 Evaluasi', '⚖️ Tradeoff', '🕘 Histori', '📐 Rumus'])

import importlib, halaman_deteksi, halaman_evaluasi, halaman_lain
importlib.reload(halaman_deteksi)
importlib.reload(halaman_evaluasi)
importlib.reload(halaman_lain)

with tab_deteksi:
    halaman_deteksi.tampilkan(MODEL, resolusi, paket, tau, W)

with tab_eval:
    halaman_evaluasi.tampilkan_evaluasi(MODEL, resolusi, tau, W)

with tab_trade:
    halaman_evaluasi.tampilkan_tradeoff(W)

with tab_hist:
    halaman_lain.tampilkan_histori(W, NAMA, NIM)

with tab_rumus:
    halaman_lain.tampilkan_rumus()

st.markdown(f"<div class='kaki'>{NAMA} · {NIM} · Tugas Matematika 3 · GLCM + Random Forest · DAGM 2007 (CC BY 4.0)</div>",
            unsafe_allow_html=True)
