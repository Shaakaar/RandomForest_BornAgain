all : bornAgain

CCC    = g++
CCOPT  = -O3
CCFLAGS= $(CCOPT)
TARGETDIR=.

OBJS2 = \
  $(TARGETDIR)/main.o \
  $(TARGETDIR)/BornAgainDecisionTree.o \
  $(TARGETDIR)/FSpace.o \
  $(TARGETDIR)/RandomForest.o

ifdef withCPLEX
  # You already moved CPLEX to C:/CPLEX, no spaces
  CPLEX_HOME = C:/CPLEX
  
  CPLEXDIR      = $(CPLEX_HOME)/cplex
  CPLEXLIBDIR   = $(CPLEXDIR)/lib/x64_win64
  CPLEXINCDIR   = $(CPLEXDIR)/include

  CONCERTDIR    = $(CPLEX_HOME)/concert
  CONCERTLIBDIR = $(CONCERTDIR)/lib/x64_win64
  CONCERTINCDIR = $(CONCERTDIR)/include

  # Extra flags: define USING_CPLEX
  CCOPT += -w -DIL_STD -DUSING_CPLEX
  CCFLAGS = $(CCOPT) -I$(CONCERTINCDIR) -I$(CPLEXINCDIR)
  CCLNDIRS  = -L$(CPLEXLIBDIR) -L$(CONCERTLIBDIR)
  CCLNFLAGS = -lconcert -lilocplex -lcplex -lpthread

  OBJS2 += $(TARGETDIR)/MIPCertificate.o
endif

$(TARGETDIR)/bornAgain: $(OBJS2)
	$(CCC) $(CCFLAGS) $(CCLNDIRS) -o $(TARGETDIR)/bornAgain $(OBJS2) $(CCLNFLAGS)

$(TARGETDIR)/%.o: %.cpp
	$(CCC) -c $(CCFLAGS) $(CPPFLAGS) -o $@ $<

clean:
	del main.o BornAgainDecisionTree.o FSpace.o RandomForest.o 2>nul
	del MIPCertificate.o 2>nul
	del bornAgain.exe 2>nul
