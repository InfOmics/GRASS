#
# Makefile by Mingz & Frank
#

CXX = g++
CUDACXX = nvcc

IDIR = include

SDIR = src
SRC = Graph.cpp Parser.cpp GraphDB.cpp DynamicBitset.cpp Multigraph.cpp Utility.cpp MatchingMachine.cpp 
CUDASRC = Matcher.cu Main.cu kernel.cu #cuPrintf.cu

ODIR = obj

INC = -I include
LIBS =

#DEFINITIONS = -D CUDA_PRINT -D PROGRESS_STATUS #-D SAVE_FILTER_REPORT -D CUDA_PRINT #-D READ_FINAL
DEFINITIONS = #-D SAVE_FILTER_REPORT -D CUDA_PRINT #-D READ_FINAL
#CUDAFLAGS = $(OPTM) -arch=sm_35 $(DEFINITIONS) #-keep -Xcompiler -rdynamic -lineinfo-lboost_system
CUDAFLAGS = $(OPTM) -arch=sm_52 $(DEFINITIONS) #-keep -Xcompiler -rdynamic -lineinfo-lboost_system  -lboost_thread-mt
CFLAGS = $(OPTM) $(DEFINITIONS) #-lboost_system -lboost_thread-mt
OPTM = -O3

EXE = GRASS

#c++ objects
_OBJS := $(addsuffix .o, $(basename $(SRC)))
OBJS = $(patsubst %,$(ODIR)/%,$(_OBJS))

$(ODIR)/%.o: $(SDIR)/%.cpp Makefile $(IDIR)/%.hpp
	$(CXX) -c $(INC) -o $@ $< $(CFLAGS)
	
#cuda objects
_CUDAOBJS := $(addsuffix .o, $(basename $(CUDASRC)))
CUDAOBJS = $(patsubst %,$(ODIR)/%,$(_CUDAOBJS))

$(ODIR)/%.o: $(SDIR)/%.cu Makefile $(IDIR)/*.h
	$(CUDACXX) -c $(INC) -o $@ $< $(CUDAFLAGS)

#build exe
$(EXE):  $(OBJS) $(CUDAOBJS)
	$(CUDACXX) -o $@ $^ $(CUDAFLAGS) $(LIBS)

clean:	flush
	rm -f $(ODIR)/*.o
	rm -f $(EXE)
	rm -f $(IDIR)/*~
	rm -f $(SDIR)/*~
	
flush:
	rm -f output/db/*
	rm -f output/query/*
	rm -f output/filterout.txt
	rm -f output/match.txt;
