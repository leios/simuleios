/*-------------multi-frame---------------------------------------------------//
 Purpose: This is for videos

//----------------------------------------------------------------------------*/

#include <stdio.h>
#include "quibble.h"

void create_filename(char *buffer, int i){
    sprintf(buffer, "check%d%d%d%d.png",
            (i%10000 - i%1000)/1000,
            (i%1000 - i%100)/100,
            (i%100 - i%10)/10,
            i%10);
}

void find_platform_device(int *platform, int *device, int argc, char **argv){

    if (argc == 1){
        fprintf(stderr, "No platform or device provided, defaulting to 0 0!\n");
    }
    else if (argc == 2){
        fprintf(stderr, "Too few arguments provided. Defaulting to 0 0!\n");
    }
    else if (argc > 3){
        fprintf(stderr, "Too many arguments provided. Defaulting to 0 0!\n");
    }
    else {
        *platform = strtol(argv[1], NULL, 10);
        *device = strtol(argv[2], NULL, 10);
        printf("Platform %d\nDevice %d\n", *platform, *device);
    }
}

int main(int argc, char **argv){

    int platform = 0;
    int device = 0;

    find_platform_device(&platform, &device, argc, argv);

    quibble_program qp = qb_parse_program_file("main.qbl");

    qb_output_program_to_file(qp, "program.txt");

    qb_configure_program(&qp, platform, device);

    int width = 1920;
    int height = 1080;
    quibble_pixels qpix = qb_create_pixel_array(qp, width, height, PRGBA8888);

    float world_size_x = 4;
    float ppu = width / world_size_x;
    float world_size_y = height / ppu;

    float world_position_x = 0;
    float world_position_y = 0;

    quibble_simple_camera qcam = qb_create_simple_camera(ppu,
                                                         world_size_x,
                                                         world_size_y,
                                                         world_position_x,
                                                         world_position_y);
    float fps = 30;

    float start_time = 0;
    float final_time = 3;
    float curr_time = start_time;
    int i = curr_time * fps;

    qb_set_args(&qp, "sdfcfs", 2,
                "quibble_pixels_prgba8888 qpix", qpix,
                "quibble_simple_camera qcam", &qcam);


    while (curr_time < final_time){
        char filename[15] = {0};
        qb_set_arg(&qp, "sdfcfs", "float time", &curr_time);

        qb_run(qp, "sdfcfs", width*height, 256);

        qb_pixels_device_to_host(qpix);

        create_filename(filename, i);
        qb_write_png_file(filename, qpix);
        printf("%s\n", filename);

        curr_time += 1/fps;
        i++;

    }

    return 0;
}

