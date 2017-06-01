# Existing variables:
#	${bindir} ../bin
#	${common_source} ../src/*c
#	${headers} ../headers
#	${common_libs} -Llib things
#	${common_flags} 


builder = gcc
nvbuilder = nvcc
linker = nvcc

#Flags to pass. Note: These are regardless of debug and release
FLAGS = 
#libs, idem
LIBS = 

#Cuda
CUC_FLAGS = -m64 -D_FORCE_INLINES --ptxas-options=-v -arch=sm_52

#Binary target file
target = ${bindir:%=%/}cuda-6
source = $(wildcard *.c) $(wildcard *.cu)
objs = $(source:.c%=.o)

#--- Common source files and headerfiles from arguments ---
# Adds *.c files from ../src
source += ${common_source}
# Adds libs from common
LIBS += ${common_libs}
# Add flags from common
FLAGS += ${common_flags}

.PHONY: all
all: release

.PHONY: release
#Release settings, adds release flags
release: FLAGS += -O3
release: CUC_FLAGS+= -O3
release: $(target)

.PHONY: debug
#debug, adds debug settings (flags)
debug: FLAGS += -g -O0
debug: CUC_FLAGS+= -G -g -O0
debug: $(target)
	
#build all object files (includes common source), headers are also included.
%.o : %.c
	$(builder) -c $(FLAGS) $< ${headers:%=-I%} -o $@ $(LIBS)
	
%.o: %.cu
	$(nvbuilder) $(CUC_FLAGS) -Xcompiler "$(FLAGS:-std%=)" -c  $< ${headers:%=-I%} -o $@ $(LIBS)
	
$(target) :
	$(linker)  $(CUC_FLAGS) -Xcompiler "$(FLAGS:-std%=)"  $(objs) ${headers:%=-I%} -o $@ $(LIBS)
	
clean:
	rm -f *.o
	rm -f $(target)
	rm -f $(target:%=%.asm)
	
asm:
	objdump -S -d $(target) >  $(target:%=%.asm)
	
