test: clean sobel
	optirun ./sobel img1.bmp

sobel: main.cpp cscbitmap.cpp sobel.cpp
	g++ -DDEBUG -o $@ $? -l OpenCL	

clean:
	rm -f sobel