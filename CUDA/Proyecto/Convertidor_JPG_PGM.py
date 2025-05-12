from PIL import Image, ImageOps
import numpy as np

def convert_to_pgm_p2_square(input_path, output_path, size=256):
    img = Image.open(input_path).convert('L')  # Escala de grises

    # Redimensionar con padding o recorte para cuadrar
    img = ImageOps.fit(img, (size, size), method=Image.LANCZOS)

    data = np.array(img)
    with open(output_path, 'w') as f:
        f.write('P2\n')
        f.write(f'# Imagen cuadrada {size}x{size}\n')
        f.write(f'{size} {size}\n')
        f.write('255\n')
        for row in data:
            f.write(' '.join(str(pixel) for pixel in row) + '\n')

# Uso:
convert_to_pgm_p2_square('lena.jpg', 'imagen_cuadrada.pgm')
