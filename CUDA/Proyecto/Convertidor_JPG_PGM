from PIL import Image
import numpy as np

def convert_to_pgm_p2(input_path, output_path):
    img = Image.open(input_path).convert('L')  # convierte a escala de grises
    data = np.array(img)
    height, width = data.shape

    with open(output_path, 'w') as f:
        f.write('P2\n')
        f.write(f'# Convertida a PGM P2\n')
        f.write(f'{width} {height}\n')
        f.write('255\n')
        for row in data:
            f.write(' '.join(str(pixel) for pixel in row) + '\n')

# Usa cualquier imagen .jpg o .png que tengas
convert_to_pgm_p2('lena.jpg', 'salida.pgm')
