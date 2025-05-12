import numpy as np
from PIL import Image
import math

# ------------------------
# Leer archivo PGM (P2)
def load_pgm(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    if lines[0].strip() != 'P2':
        raise ValueError("Formato PGM no soportado")

    width, height = map(int, lines[2].strip().split())
    max_val = int(lines[3].strip())

    data = []
    for line in lines[4:]:
        data.extend(map(int, line.split()))

    return np.array(data, dtype=float).reshape((height, width)), width, height


# ------------------------
# Guardar archivo PGM
def save_pgm(filename, data):
    height, width = data.shape
    max_val = 255
    with open(filename, 'w') as file:
        file.write("P2\n")
        file.write(f"{width} {height}\n")
        file.write(f"{max_val}\n")
        for i in range(height):
            file.write(" ".join(map(str, data[i].astype(int))) + "\n")


# ------------------------
# Función para encontrar el siguiente número que sea una potencia de 2
def next_power_of_two(x):
    return 2**math.ceil(math.log2(x))


# ------------------------
# Rellenar imagen para que sea de tamaño potencia de 2
def pad_image(data, width, height):
    new_width = next_power_of_two(width)
    new_height = next_power_of_two(height)

    padded = np.zeros((new_height, new_width), dtype=float)
    padded[:height, :width] = data

    print(f"Dimensiones ajustadas a: {new_width} x {new_height} (potencias de 2)")
    return padded, new_width, new_height


# ------------------------
# Calcular magnitud y normalizar
def calculate_magnitude_and_normalize(real, imag, width, height):
    mag = np.sqrt(real**2 + imag**2)
    log_mag = np.log1p(mag)  # Logaritmo de 1 + magnitud

    max_val = np.max(log_mag)
    if max_val > 0:
        log_mag = (log_mag / max_val) * 255
    return log_mag


# ------------------------
# Aplicar FFT Shift
def apply_fft_shift(data, width, height):
    half_w = width // 2
    half_h = height // 2

    shifted = np.copy(data)
    shifted[:half_h, :half_w], shifted[half_h:, half_w:] = shifted[half_h:, half_w:], shifted[:half_h, :half_w]
    shifted[:half_h, half_w:], shifted[half_h:, :half_w] = shifted[half_h:, :half_w], shifted[:half_h, half_w:]

    return shifted


# ------------------------
# FFT 1D en filas
def fft_1d_row(data, width, height):
    real = np.zeros_like(data)
    imag = np.zeros_like(data)

    for y in range(height):
        real[y, :] = np.real(np.fft.fft(data[y, :]))
        imag[y, :] = np.imag(np.fft.fft(data[y, :]))

    return real, imag


# ------------------------
# FFT 1D en columnas
def fft_1d_col(data, width, height):
    real = np.zeros_like(data)
    imag = np.zeros_like(data)

    for x in range(width):
        real[:, x] = np.real(np.fft.fft(data[:, x]))
        imag[:, x] = np.imag(np.fft.fft(data[:, x]))

    return real, imag


# ------------------------
# IFFT 1D en filas
def ifft_1d_row(real, imag, width, height):
    data = np.zeros_like(real)

    for y in range(height):
        data[y, :] = np.real(np.fft.ifft(real[y, :] + 1j * imag[y, :]))

    return data


# ------------------------
# IFFT 1D en columnas
def ifft_1d_col(real, imag, width, height):
    data = np.zeros_like(real)

    for x in range(width):
        data[:, x] = np.real(np.fft.ifft(real[:, x] + 1j * imag[:, x]))

    return data


# ------------------------
# Función principal
def main():
    input_file = "barbara.ascii.pgm"
    output_fft = "resultado.pgm"
    output_reconstructed = "reconstruida.pgm"

    # Cargar imagen
    data, width, height = load_pgm(input_file)
    print(f"Dimensiones originales: {width} x {height}")

    # Ajustar tamaño a potencia de 2
    padded_data, new_width, new_height = pad_image(data, width, height)

    # Realizar FFT
    real_out, imag_out = fft_1d_row(padded_data, new_width, new_height)
    real_out, imag_out = fft_1d_col(real_out, new_width, new_height)

    # Calcular magnitud y normalizar
    magnitude = calculate_magnitude_and_normalize(real_out, imag_out, new_width, new_height)

    # Aplicar FFT shift
    shifted_magnitude = apply_fft_shift(magnitude, new_width, new_height)

    # Guardar imagen FFT
    save_pgm(output_fft, shifted_magnitude)
    print(f"Imagen de la magnitud FFT guardada como {output_fft}")

    # Realizar IFFT
    ifft_result = ifft_1d_col(real_out, imag_out, new_width, new_height)
    ifft_result = ifft_1d_row(ifft_result, imag_out, new_width, new_height)

    # Guardar imagen reconstruida
    save_pgm(output_reconstructed, ifft_result)
    print(f"Imagen reconstruida guardada como {output_reconstructed}")


if __name__ == "__main__":
    main()
