import numpy as np
from PIL import Image
import math
import time

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
def main():
    input_file = "barbara.ascii.pgm"
    output_fft = "resultado.pgm"
    output_reconstructed = "reconstruida.pgm"

    start_total = time.time()

    # Cargar imagen
    data, width, height = load_pgm(input_file)
    print(f"Dimensiones originales: {width} x {height}")

    # Ajustar tamaño a potencia de 2
    padded_data, new_width, new_height = pad_image(data, width, height)

    # ------------------------
    # FFT directa
    start_fft = time.time()
    real_out, imag_out = fft_1d_row(padded_data, new_width, new_height)
    real_out, imag_out = fft_1d_col(real_out, new_width, new_height)
    end_fft = time.time()
    print(f"Tiempo FFT directa: {(end_fft - start_fft)*1000:.2f} ms")

    # ------------------------
    # Magnitud y FFT shift
    start_mag = time.time()
    magnitude = calculate_magnitude_and_normalize(real_out, imag_out, new_width, new_height)
    shifted_magnitude = apply_fft_shift(magnitude, new_width, new_height)
    save_pgm(output_fft, shifted_magnitude)
    end_mag = time.time()
    print(f"Tiempo magnitud + shift + guardado: {(end_mag - start_mag)*1000:.2f} ms")
    print(f"Imagen de la magnitud FFT guardada como {output_fft}")

    # ------------------------
    # IFFT (reconstrucción)
    start_ifft = time.time()
    ifft_result = ifft_1d_col(real_out, imag_out, new_width, new_height)
    ifft_result = ifft_1d_row(ifft_result, imag_out, new_width, new_height)
    save_pgm(output_reconstructed, ifft_result)
    end_ifft = time.time()
    print(f"Tiempo IFFT + guardado: {(end_ifft - start_ifft)*1000:.2f} ms")
    print(f"Imagen reconstruida guardada como {output_reconstructed}")

    end_total = time.time()
    print(f"Tiempo total del proceso: {(end_total - start_total)*1000:.2f} ms")


if __name__ == "__main__":
    main()
