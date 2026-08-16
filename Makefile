.PHONY: reproduce smoke clean

PYTHON ?= python3

reproduce:
	$(PYTHON) reproduce.py

smoke:
	$(MAKE) -C src
	./src/mc_canonical -L 8 -N 8 -T 0.3 -therm 1000 -nmeas 10000 -nbin 20 -seed 42

clean:
	$(MAKE) -C src clean
	rm -rf results
