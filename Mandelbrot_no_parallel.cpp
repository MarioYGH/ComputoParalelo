
#include <stdio.h>
#include <stdlib.h>
#include <time.h>   // Para medir tiempo con clock()

/*
   Parámetros de la imagen:
   (puedes modificarlos para probar diferentes tamaños y niveles de detalle)
*/
#define WIDTH    800   // Ancho de la imagen en píxeles
#define HEIGHT   600   // Alto de la imagen en píxeles
#define MAX_ITER 1000  // Iteraciones máximas para cada punto

/*
   Límites del plano complejo donde se representa el fractal
   (se pueden ajustar para hacer zoom o moverse a diferentes regiones)
*/
#define XMIN -2.0
#define XMAX  1.0
#define YMIN -1.2
#define YMAX  1.2

/*
   Función que calcula cuántas iteraciones tarda un punto (cx, cy)
   en salirse del rango (se considera escape cuando el valor de
   z = x + i*y supere el módulo 2).
*/
int mandelbrot_iterations(double cx, double cy) {
    double x = 0.0;
    double y = 0.0;
    int iter = 0;

    while ((x*x + y*y <= 4.0) && (iter < MAX_ITER)) {
        double xTemp = x*x - y*y + cx;
        y = 2.0*x*y + cy;
        x = xTemp;
        iter++;
    }
    return iter;
}

int main(void) {
    // Arreglo para guardar los valores de iteración de cada píxel
    int *iterations = (int *)malloc(WIDTH * HEIGHT * sizeof(int));
    if (!iterations) {
        fprintf(stderr, "Error al asignar memoria.\n");
        return 1;
    }

    // Medir el tiempo de inicio (serial)
    clock_t start = clock();

    // Ciclo doble para calcular el fractal de Mandelbrot (serial)
    for (int py = 0; py < HEIGHT; py++) {
        for (int px = 0; px < WIDTH; px++) {
            // Calcular la coordenada en el plano complejo correspondiente al píxel (px, py)
            double cx = XMIN + (XMAX - XMIN) * (double)px / (WIDTH - 1);
            double cy = YMIN + (YMAX - YMIN) * (double)py / (HEIGHT - 1);

            // Calcular el número de iteraciones para este punto
            int iter = mandelbrot_iterations(cx, cy);

            // Guardar el resultado en el arreglo (índice lineal)
            iterations[py * WIDTH + px] = iter;
        }
    }

    // Medir el tiempo de fin (serial)
    clock_t end = clock();
    double elapsed_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tiempo de cálculo: %f segundos\n", elapsed_time);

    // Crear archivo PPM en modo texto (P3)
    FILE *fp = fopen("mandelbrot.ppm", "w");
    if (!fp) {
        fprintf(stderr, "Error al crear el archivo PPM.\n");
        free(iterations);
        return 1;
    }

    // Encabezado del PPM (formato P3, anchura, altura, valor máximo de color)
    fprintf(fp, "P3\n%d %d\n255\n", WIDTH, HEIGHT);

    // Escribir los datos de color (RGB por píxel).
    // Se usa una simple gradación de color según el número de iteraciones.
    for (int py = 0; py < HEIGHT; py++) {
        for (int px = 0; px < WIDTH; px++) {
            int iter = iterations[py * WIDTH + px];

            // Mapear iteraciones a un color simple (por ejemplo, degradado).
            int r = (iter % 256);
            int g = (iter * 5) % 256;
            int b = (iter * 13) % 256;

            // Si el punto está en el conjunto (iter == MAX_ITER),
            // lo pintamos de negro (0,0,0) o un color distintivo.
            if (iter == MAX_ITER) {
                r = 0;
                g = 0;
                b = 0;
            }

            fprintf(fp, "%d %d %d ", r, g, b);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    free(iterations);

    printf("Imagen generada en \"mandelbrot.ppm\".\n");
    return 0;
}
