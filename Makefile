# Makefile for LINUX only
MACHTYPE=$(shell uname -m)

GCC			= g++

CC=gcc
# OPTFLAGS=-g2
OPTFLAGS=-O3 -fopenmp -ffast-math -w -mfpmath=sse -msse2 -funroll-loops 
OTHERFLAGS=-w -fopenmp -pthread

INCLUDE_PATH=-IThirdParty/ann_1.1/include/ANN -IThirdParty/ann_1.1_char/include/ANN -IThirdParty/zlib/include
LIBS=ThirdParty/libANN.a ThirdParty/libANN_char.a 

OBJS=main.o keys2.o kmeans.o kmeans_kd.o qsort.o util.o VocabFlatNode.o VocabTree.o VocabTreeBuild.o VocabTreeIO.o VocabTreeUtil.o

CPPFLAGS=$(INCLUDE_PATH) $(LIB_PATH) $(OTHERFLAGS) $(OPTFLAGS)

BIN=VocabTree

all: $(BIN)

$(BIN): $(OBJS)
	g++ -o $(CPPFLAGS) -o $(BIN) $(OBJS) $(LIBS)

clean:
	rm -f *.o *~ $(LIB)
