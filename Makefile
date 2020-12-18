CC = gcc
LIBS = -lm -ldl

all: 
	$(CC) $(LIBS) -o CurveSubdivision CurveSubdivision.c 

clean:
	$(RM) CurveSubdivision *.txt *.ps

run:
	./CurveSubdivision
