import requests
import os
import getpass
import pandas as pd

# Get credentials of the CEFCA portal
user = input('User: ')
password = getpass.getpass()

session = requests.Session()
session.auth = (user, password)

auth = session.post('https://archive.cefca.es/')

# miniJPAS
print('\nminiJPAS\n')
tile_ids = pd.read_csv('csv/minijpas.TileImage.csv', header=1)['TILE_ID']

foldername = '/home/alberto/almacen/images_fits/minijpas/PSF_model'
os.makedirs(foldername, exist_ok=True)

for tile in tile_ids:
    print(f'Downloading: {tile}')
    url = f'http://archive.cefca.es/catalogues/vo/siap/minijpas-pdr201912/get_psf_file?id={tile}'
    response = session.get(url)
    with open(f'{foldername}/PSF_{tile}.fits', 'wb') as f:
        f.write(response.content)

# J-NEP
print('\nJ-NEP\n')
tile_ids = pd.read_csv('csv/jnep.TileImage.csv', header=1)['TILE_ID']

foldername = '/home/alberto/almacen/images_fits/jnep/PSF_model'
os.makedirs(foldername, exist_ok=True)

for tile in tile_ids:
    print(f'Downloading: {tile}')
    url = f'http://archive.cefca.es/catalogues/vo/siap/jnep-pdr202107/get_psf_file?id={tile}'
    response = session.get(url)
    with open(f'{foldername}/PSF_{tile}.fits', 'wb') as f:
        f.write(response.content)