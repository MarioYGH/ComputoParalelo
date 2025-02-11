#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "stb_image.h"
#include "stb_image_write.h"



//Transformación de una imagen en escala de grises (real)
void transformar_imagen_real(const char *input_filename, const char *output_filename) {
    int width, height, channels;
    printf("Image path: %s \n", input_filename);
    unsigned char *imagen = stbi_load(input_filename, &width, &height, &channels, 0);

    if (!imagen) {
        printf("Error al cargar la imagen.\n");
        return;
    }

    int size = width * height * channels;
    unsigned char *gris = (unsigned char *)malloc(width * height);

#pragma omp parallel for
    for (int i = 0; i < width * height; i++) {
        int idx = i * channels;
        gris[i] = (unsigned char)(0.3 * imagen[idx] + 0.59 * imagen[idx + 1] + 0.11 * imagen[idx + 2]);
    }

    stbi_write_png(output_filename, width, height, 1, gris, width);

    printf("Imagen transformada a escala de grises guardada en '%s'\n", output_filename);

    stbi_image_free(imagen);
    free(gris);
}

int main() {

    transformar_imagen_real("Example.jpg", "Example_gris.png"); //Si no se encuentra en el mismo archivo debes de poner todo el path // /Users/rayoy/Documents/Visualstudio/RGB_toBW/
    
    return 0;
}
