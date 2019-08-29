CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
LFLAG = -lgsl -lm -lgslcblas
INCLUDES = -I/usr/include -I/data/toni/eigen-3.3.4 -I/data/toni/gsl-2.4-install/include
LIBS = -L/usr/lib/x86_64-linux-gnu -L/data/toni/gsl-2.4-install/lib

embed : ransampl.o linelib.o main.o
	$(CC) $(CFLAGS) -o embed ransampl.o linelib.o main.o $(INCLUDES) $(LIBS) $(LFLAG)

ransampl.o : ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS) $(LFLAG)

linelib.o : linelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c linelib.cpp $(INCLUDES) $(LIBS) $(LFLAG)

main.o : main.cpp linelib.o
	$(CC) $(CFLAGS) -c main.cpp $(INCLUDES) $(LIBS) $(LFLAG)

clean :
	rm -rf *.o embed

	