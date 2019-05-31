CFLAGS ?= -march=native -O3 -DNDEBUG -Wall -Wno-unused
LDLIBS += -ljpeg -ltiff -lpng -lz -lfftw3f -lm

OBJ = src/iio.o
BIN = plambda iion awgn

default: $(BIN)

% : src/%.o $(OBJ)
	$(CC) $(LDFLAGS) -o $@ $^ $(LDLIBS)

clean: ; @$(RM) $(BIN) src/*.o
.PHONY: default clean

# hack for older compilers (adds gnu99 option to CFLAGS, if necessary)
ifeq (,$(shell $(CC) $(CFLAGS) -dM -E - < /dev/null | grep __STDC_VERSION__))
CFLAGS := $(CFLAGS) -std=gnu99
endif
