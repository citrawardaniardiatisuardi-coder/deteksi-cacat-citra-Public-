"""Tab Deteksi: pilih/unggah gambar → proses → heatmap → vonis."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import streamlit as st

import inti


def peta_warna(W):
    return LinearSegmentedColormap.from_list('tema', [W['krem'], W['lembut'], W['aksen'], W['utama'], W['tua']])


def campur(abu_u8, heat, cmap, alpha=0.65):
    """Alpha blending: hasil = (1 − a·h)·abu + a·h·warna(h)  → kombinasi konveks dua warna."""
    g = np.repeat(abu_u8[..., None] / 255.0, 3, axis=2)
    a = (alpha * heat)[..., None]
    return np.clip((1 - a) * g + a * cmap(heat)[..., :3], 0, 1)


def hex2rgb(kode):
    """'#D9A441' → (0.85, 0.64, 0.25): tiap 2 digit heksadesimal = 0..255, lalu dibagi 255."""
    return np.array([int(kode[i:i + 2], 16) for i in (1, 3, 5)]) / 255


def dua_warna(mask_u8, W):
    """Mask hitam-putih → dua warna tema (latar cokelat tua, area cacat warna aksen)."""
    m = (mask_u8 > 127)[..., None]
    return np.where(m, hex2rgb(W['aksen']), hex2rgb(W['tua']))


def sorot(abu_u8, m, rgb, alpha=0.55):
    """Area m (0..1) diberi warna rgb secara transparan: (1 − a·m)·abu + a·m·rgb."""
    g = np.repeat(abu_u8[..., None] / 255.0, 3, axis=2)
    a = (alpha * m)[..., None]
    return np.clip((1 - a) * g + a * rgb, 0, 1)


def judul(no, teks):
    st.markdown(f"<div class='subjudul'><b>{no}</b>{teks}</div>", unsafe_allow_html=True)


def keterangan(teks):
    st.markdown(f"<div class='cap'>{teks}</div>", unsafe_allow_html=True)


def matriks_peluang(p, W, cmap):
    fig, ax = plt.subplots(figsize=(4, 4), facecolor=W['krem'])
    M = p.reshape(inti.GRID, inti.GRID)
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1)
    for i in range(inti.GRID):
        for j in range(inti.GRID):
            ax.text(j, i, f'{M[i, j]:.2f}'[1:] if M[i, j] < 1 else '1', ha='center', va='center', fontsize=8,
                    color=W['krem'] if M[i, j] > 0.5 else W['tua'])
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(W['tua'])
    ax.set_title('Matriks peluang pₖ (8×8)', color=W['utama'], fontweight='bold')
    fig.tight_layout()
    return fig


def pilih_gambar():
    sumber = st.radio('Sumber gambar', ['📁 Contoh data uji', '⬆️ Unggah sendiri'], horizontal=True)
    if sumber.startswith('📁'):
        if not os.path.exists('contoh/daftar.csv'):
            st.warning('Folder contoh belum ada. Jalankan sel 15a di notebook.')
            return None, None, None, None
        D = pd.read_csv('contoh/daftar.csv')
        i = st.selectbox('Pilih gambar', D.index,
                         format_func=lambda i: f"{D.loc[i, 'file']}  ·  {'CACAT' if D.loc[i, 'label'] == 1 else 'normal'}")
        r = D.loc[i]
        mask = None
        if isinstance(r['mask'], str) and r['mask']:
            mask = inti.baca_abu(os.path.join('contoh', r['mask']))
        return inti.baca_abu(os.path.join('contoh', r['file'])), r['file'], mask, int(r['label'])
    up = st.file_uploader('Pilih gambar JPG/PNG (abu-abu atau berwarna)', type=['png', 'jpg', 'jpeg'])
    if up is None:
        return None, None, None, None
    return inti.baca_abu(up.getvalue()), up.name, None, None


def tampilkan(MODEL, resolusi, paket, tau, W):
    cmap = peta_warna(W)
    img, nama, mask, label = pilih_gambar()
    if img is None:
        st.markdown("<div class='kartu kosong'><div class='ikon'>🖼️</div><h3>Belum ada gambar</h3>"
                    "Pilih contoh atau unggah gambar dulu.</div>", unsafe_allow_html=True)
        return

    h = inti.proses_citra(img, resolusi)
    p = inti.prediksi(paket['model'], h['fitur'], tau)
    heat = inti.peta_panas(p['prob_petak'], resolusi)

    # ---------- 1. Citra asli vs hasil proses ----------
    judul(1, 'Citra asli vs hasil proses')
    kol = st.columns(5)
    tampil = [(img, f'Asli {img.shape[1]}×{img.shape[0]}'),
              (h['resize'], f'Resize {resolusi}×{resolusi}'),
              (h['normal'], 'Normalisasi min-max'),
              (cmap(h['peta_D'] / 255.0)[..., :3], 'Peta kekasaran |ln(s/median)|'),
              (dua_warna(h['mask_seg'], W), f'Segmentasi Otsu (t = {h["ambang_otsu"]:.0f})')]
    for k, (g, cap) in zip(kol, tampil):
        with k:
            st.image(g, width='stretch', clamp=True)
            keterangan(cap)

    # ---------- 2. Heatmap & vonis ----------
    judul(2, 'Heatmap peluang cacat & vonis')
    k1, k2, k3 = st.columns([1.15, 1.15, 1])
    with k1:
        st.image(campur(h['normal'], heat, cmap), width='stretch', clamp=True)
        keterangan('Heatmap p(cacat) di atas citra')
    with k2:
        if mask is not None:
            m_n = inti.resize_citra(mask, resolusi) / 255.0
            st.image(sorot(h['normal'], m_n, hex2rgb(W['aksen'])), width='stretch', clamp=True)
            keterangan('Label asli (elips DAGM) di atas citra')
        else:
            fig = matriks_peluang(p['prob_petak'], W, cmap)
            st.pyplot(fig, width='stretch')
            plt.close(fig)
    with k3:
        if p['cacat']:
            st.markdown(f"<div class='vonis cacat'>⚠️ CACAT<small>S = {p['skor']:.3f} ≥ τ = {tau:.2f}</small></div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='vonis ok'>✅ NORMAL<small>S = {p['skor']:.3f} &lt; τ = {tau:.2f}</small></div>",
                        unsafe_allow_html=True)
        k_maks = int(np.argmax(p['prob_petak']))
        baris = [('Petak tercurigai', f'baris {k_maks // inti.GRID + 1}, kolom {k_maks % inti.GRID + 1}'),
                 ('Waktu proses', f"{h['waktu_ms']:.0f} ms")]
        if label is not None:
            benar = (label == 1) == p['cacat']
            baris.append(('Label sebenarnya', ('Cacat' if label == 1 else 'Normal') +
                          (' · prediksi benar ✓' if benar else ' · prediksi salah ✗')))
        isi = ''.join(f"<div class='baris-info'><span>{a}</span><b>{b}</b></div>" for a, b in baris)
        st.markdown(f"<div class='kartu'>{isi}</div>", unsafe_allow_html=True)

    # ---------- 3. Detail matematika ----------
    with st.expander(f'🔬 Fitur GLCM petak paling mencurigakan (petak #{k_maks + 1})'):
        df = pd.DataFrame({'fitur': inti.NAMA_FITUR, 'nilai': h['fitur'][k_maks],
                           'kepentingan RF': paket['model'].feature_importances_})
        st.dataframe(df.sort_values('kepentingan RF', ascending=False).round(4), width='stretch', hide_index=True)

    if st.toggle('⚖️ Bandingkan gambar ini di semua resolusi'):
        rows = []
        for n, pk in MODEL.items():
            hn = inti.proses_citra(img, n)
            pn = inti.prediksi(pk['model'], hn['fitur'], pk['tau'])
            rows.append({'resolusi': f'{n}×{n}', 'τ*': pk['tau'], 'skor S': round(pn['skor'], 3),
                         'vonis': '⚠️ CACAT' if pn['cacat'] else '✅ normal', 'waktu (ms)': round(hn['waktu_ms'])})
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)

    # simpan hasil terakhir (dipakai tab Histori & ekspor laporan nanti)
    st.session_state['terakhir'] = {'file': nama, 'resolusi': resolusi, 'tau': tau, 'skor': p['skor'],
                                    'prediksi': 'cacat' if p['cacat'] else 'normal',
                                    'label_asli': '' if label is None else ('cacat' if label == 1 else 'normal'),
                                    'waktu_ms': h['waktu_ms']}
