import requests
import os
import getpass

# Get credentials of the CEFCA portal
user = input('User: ')
password = getpass.getpass()

session = requests.Session()
session.auth = (user, password)

auth = session.post('https://archive.cefca.es/')

# miniJPAS
print('\nminiJPAS\n')
tile_ids = [2241, 2243, 2406, 2470]
filter_ids = [i + 1 for i in range(60)]

foldername = '/home/alberto/almacen/images_fits/minijpas'
os.makedirs(foldername, exist_ok=True)

for tile in tile_ids:
    for filter in filter_ids:
        print(f'Downloading: {tile}-{filter}')
        url = f'http://archive.cefca.es/catalogues/vo/siap/minijpas-pdr201912/get_fits?id={tile}&filter={filter}'
        response = session.get(url)
        open(f'{foldername}/{tile}-{filter}.fits', 'wb').write(response.content)

# J-NEP
print('\nJ-NEP\n')
tile_ids = [2520]
filter_ids = [i + 1 for i in range(60)]

foldername = '/home/alberto/almacen/images_fits/jnep'
os.makedirs(foldername, exist_ok=True)

for tile in tile_ids:
    for filter in filter_ids:
        print(f'Downloading: {tile}-{filter}')
        url = f'https://archive.cefca.es/catalogues/vo/siap/jnep-pdr202107/get_fits?id={tile}&filter={filter}'
        response = session.get(url)
        open(f'{foldername}/{tile}-{filter}.fits', 'wb').write(response.content)