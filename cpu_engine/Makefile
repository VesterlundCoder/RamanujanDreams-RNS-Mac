CC ?= clang
BUILD_DIR ?= build
CFLAGS_COMMON := -std=c11 -O3 -Wall -Wextra -Wshadow -Wconversion -Wno-sign-conversion -pthread -Iinclude
CFLAGS_PRECISE := $(CFLAGS_COMMON) -fno-fast-math -ffp-contract=off
LDFLAGS_COMMON := -pthread -lm

UNAME_M := $(shell uname -m)
ifeq ($(UNAME_M),arm64)
  CFLAGS_COMMON += -mcpu=native
  CFLAGS_PRECISE += -mcpu=native
endif
ifeq ($(UNAME_M),aarch64)
  CFLAGS_COMMON += -mcpu=native
  CFLAGS_PRECISE += -mcpu=native
endif

.PHONY: all clean test

all: $(BUILD_DIR)/dreams_fixed_start_cpu $(BUILD_DIR)/matmul6_bench $(BUILD_DIR)/rns_matmul6_bench

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/cmf_cpu.o: src/cmf_cpu.c include/cmf_cpu.h include/df64.h | $(BUILD_DIR)
	$(CC) $(CFLAGS_PRECISE) -c $< -o $@

$(BUILD_DIR)/dreams_fixed_start_cpu: src/dreams_fixed_start_cpu.c $(BUILD_DIR)/cmf_cpu.o | $(BUILD_DIR)
	$(CC) $(CFLAGS_PRECISE) $^ -o $@ $(LDFLAGS_COMMON)

$(BUILD_DIR)/matmul6_bench: src/matmul6_bench.c | $(BUILD_DIR)
	$(CC) $(CFLAGS_COMMON) $< -o $@ $(LDFLAGS_COMMON)

$(BUILD_DIR)/rns_matmul6_bench: src/rns_matmul6_bench.c | $(BUILD_DIR)
	$(CC) $(CFLAGS_COMMON) $< -o $@ $(LDFLAGS_COMMON)

test: all
	python3 scripts/gen_dirs.py --count 32 --output $(BUILD_DIR)/test_dirs.bin --seed 1 --nonzero 1
	$(BUILD_DIR)/dreams_fixed_start_cpu $(BUILD_DIR)/test_dirs.bin $(BUILD_DIR)/test_cpu.bin "1,1,1,1,1,1,2,2,2,2,2" 5 4 -1 1 4 32 2 4
	python3 scripts/inspect_fsdf64.py $(BUILD_DIR)/test_cpu.bin
	$(BUILD_DIR)/matmul6_bench --n 1000 --repeats 1 --threads 2
	$(BUILD_DIR)/rns_matmul6_bench --pairs 100 --steps 2 --threads 2

clean:
	rm -rf $(BUILD_DIR)
