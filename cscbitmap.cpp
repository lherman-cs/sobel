// Code by Bidur Bohara (LSU) in collaboration with Brygg Ullmer

#include "cscbitmap.h"
#include <stdio.h>

Bitmap::Bitmap()
{
    bmpWidth = 0;
    bmpHeight = 0;
    bmpSize = 0;
    offset = 0;
    bitsPerPixel = 0;
}

///
unsigned char* Bitmap::readGrayBitmap(const char *file)
{
    // Open bitmap file to read
    FILE *fp = fopen(file, "rb");
    if(!fp)
    {
        printf("Error! Cannot open input file.\n");
        return 0;
    }

    unsigned int status = 0;
    status = fseek(fp, 10, SEEK_SET); // Seek to width
    status = fread((void*)&offset, sizeof(unsigned int), 1, fp);

    status = fseek(fp, 18, SEEK_SET); // Seek to width
    status = fread((void*)&bmpWidth, sizeof(int), 1, fp);

    status = fseek(fp, 22, SEEK_SET); // Seek to height
    status = fread((void*)&bmpHeight, sizeof(int), 1, fp);

    status = fseek(fp, 28, SEEK_SET); // Seek to bits per pixel
    status = fread((void*)&bitsPerPixel, sizeof(unsigned short), 1, fp);

    status = fseek(fp, 34, SEEK_SET); // Seek to bitmap image size
    status = fread((void*)&bmpSize, sizeof(unsigned int), 1, fp);

    /// Read the Bitmap Header info.
    bmpHeader = new unsigned char[offset];
    status = fseek(fp, 0, SEEK_SET);
    status = fread((void*)bmpHeader, sizeof(unsigned char), offset, fp);

    /// Read the Bitmap image data.
    unsigned char* bmpData = new unsigned char[bmpSize];

    /// Seek to the position of image data.
    status = fseek(fp, offset, SEEK_SET);
    status = fread(bmpData, sizeof(unsigned char), bmpSize, fp);
    //bmpSize = status > 0 ? status : bmpSize;

    fclose(fp);
    return bmpData;
}


///
void Bitmap::writeGrayBmp(unsigned char* data)
{
    FILE *wp = fopen("1.bmp", "wb");

    if(!data)
        printf("No data to be written!!!");

    unsigned int status = 0;

    status = fwrite((const void*)bmpHeader, sizeof(unsigned char),
                    offset, wp);
    status = fwrite((const void*)data, sizeof(unsigned char), bmpSize, wp);
    fclose(wp);
}

