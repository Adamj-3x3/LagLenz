# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=[],
    datas=[('/Users/adameminger/Desktop/PythonLagLens/LagLenz/data/Bitcoin Price.csv', 'data'), ('/Users/adameminger/Desktop/PythonLagLens/LagLenz/data/Chainlink Price.csv', 'data'), ('/Users/adameminger/Desktop/PythonLagLens/LagLenz/data/DJI Price.csv', 'data'), ('/Users/adameminger/Desktop/PythonLagLens/LagLenz/data/DXY.csv', 'data'), ('/Users/adameminger/Desktop/PythonLagLens/LagLenz/data/Global Liquity.csv', 'data'), ('/Users/adameminger/Desktop/PythonLagLens/LagLenz/data/NASDAQ Price.csv', 'data'), ('/Users/adameminger/Desktop/PythonLagLens/LagLenz/data/US10Y.csv', 'data')],
    hiddenimports=['matplotlib.backends.backend_tkagg'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
app = BUNDLE(
    exe,
    name='main.app',
    icon=None,
    bundle_identifier=None,
)
