sobel: main.cpp cscbitmap.cpp sobel.cpp
	g++ -Ofast -o $@ $? -l OpenCL	

clean:
	rm -f sobel