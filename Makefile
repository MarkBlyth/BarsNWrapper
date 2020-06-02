CC=gcc
CFLAGS=-ansi

com.o: com.c
	$(CC) -c com.c

ranlibsub.o: ranlibsub.c ranlib.h
	$(CC) -c ranlibsub.c

logspline.o: logspline.c logspline.h
	$(CC) -c logspline.c

barsN_utils.o: barsN_utils.c barsN_utils.h
	$(CC) -c barsN_utils.c

barsN_funcs.o: barsN_funcs.c barsN_funcs1.h barsN_funcs.h
	$(CC) -c barsN_funcs.c

barsN.out: barsN.c barsN_funcs.h barsN_funcs.o barsN_utils.o logspline.o ranlibsub.o com.o
	$(CC) -o barsN.out $(CFLAGS) barsN.c barsN_funcs.o barsN_utils.o logspline.o ranlibsub.o com.o -llapack -lblas -lgfortran -lm -ldl


