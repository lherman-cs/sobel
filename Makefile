sobel: main.cpp cscbitmap.cpp sobel.cpp
	g++ -o $@ $? -l OpenCL	

test: clean sobel
	optirun ./sobel img1.bmp

clean:
	rm -f sobel