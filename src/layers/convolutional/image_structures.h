
typedef struct ImageArray
{
    double *pixels;
    int m;
    int n;
} ImageArray;

typedef struct ImageLayer
{
    ImageArray **images;
    int num_images;
} ImageLayer;
