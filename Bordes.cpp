#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "stb_image.h"
#include "stb_image_write.h"

// Matrices (kernels) de detección de bordes Sobel
int Gx[3][3] = {
    {-1,  0,  1},
    {-2,  0,  2},
    {-1,  0,  1}
};

int Gy[3][3] = {
    {-1, -2, -1},
    { 0,  0,  0},
    { 1,  2,  1}
};

// Función para aplicar la detección de bordes usando Sobel
void deteccion_bordes(const char *input_filename, const char *output_filename) {
    int width, height, channels;
    printf("Cargando imagen: %s\n", input_filename);
    unsigned char *imagen = stbi_load(input_filename, &width, &height, &channels, 0);

    if (!imagen) {
        printf("Error al cargar la imagen.\n");
        return;
    }

    // Convertimos la imagen a escala de grises
    unsigned char *gris = (unsigned char *)malloc(width * height);
    #pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;
        gris[i] = (unsigned char)(0.3 * imagen[idx] + 0.59 * imagen[idx + 1] + 0.11 * imagen[idx + 2]);
    }

    // Aplicamos la convolución para detección de bordes
    unsigned char *bordes = (unsigned char *)malloc(width * height);
    
    #pragma omp parallel for collapse(2)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sumX = 0;
            int sumY = 0;

            // Aplicamos los kernels de Sobel
            for (int i = -1; i <= 1; i++) {
                for (int j = -1; j <= 1; j++) {
                    int pixel = gris[(y + i) * width + (x + j)];
                    sumX += Gx[i + 1][j + 1] * pixel;
                    sumY += Gy[i + 1][j + 1] * pixel;
                }
            }

            // Magnitud del gradiente
            int magnitud = abs(sumX) + abs(sumY);
            if (magnitud > 255) magnitud = 255;
            bordes[y * width + x] = (unsigned char)magnitud;
        }
    }

    // Guardamos la imagen resultante en escala de grises
    stbi_write_png(output_filename, width, height, 1, bordes, width);

    printf("Imagen de detección de bordes guardada en '%s'\n", output_filename);

    // Liberamos memoria
    stbi_image_free(imagen);
    free(gris);
    free(bordes);
}

int main() {
    deteccion_bordes("Example.jpg", "Example_bordes.png");
    return 0;
}
