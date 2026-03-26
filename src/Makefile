# Makefile for n-queens lattice gas Monte Carlo simulation

CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -lm

TARGET = mc_canonical

all: $(TARGET)

$(TARGET): mc_canonical.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
