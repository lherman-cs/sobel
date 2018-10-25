// Code by Bidur Bohara (LSU) in collaboration with Brygg Ullmer

#ifndef BITMAP_H
#define BITMAP_H

class Bitmap
{
public:
    Bitmap();

    unsigned int bmpWidth;
    unsigned int bmpHeight;
    unsigned int bmpSize;

    unsigned char* readGrayBitmap(const char*file);
    void writeGrayBmp(unsigned char* data);

private:
    unsigned char* bmpHeader;
    unsigned int offset;
    unsigned int bitsPerPixel;

};

#endif // BITMAP_H
