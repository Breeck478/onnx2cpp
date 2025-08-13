SUBDIRS := $(filter-out CMakeFiles,$(wildcard */))

MAKE_SUBDIRS := $(shell for d in $(SUBDIRS); do \
    if [ -f $$d/Makefile ]; then echo $$d; fi; \
done)

.PHONY: all run_all $(MAKE_SUBDIRS) clean ../Testing/dcoTests/Passed.txt ../Testing/dcdcoTestso/NotPassed.txt

all: run_all

clean: 
	@echo "Cleaning all executables in subdirectories..."
	@for d in $(MAKE_SUBDIRS); do \
		$(MAKE) -C $$d clean; \
	done

run_all: ../Testing/dcoTests/Passed.txt ../Testing/dcoTests/NotPassed.txt $(MAKE_SUBDIRS)

../Testing/dcoTests/Passed.txt ../Testing/dcoTests/NotPassed.txt:
	@mkdir -p ../Testing/dcoTests
	@rm -f ../Testing/dcoTests/Passed.txt ../Testing/dcoTests/NotPassed.txt

$(MAKE_SUBDIRS):
	@echo "Running make in $@"
	@$(MAKE) -C $@ > $@.log 2>&1; \
	if [ $$? -eq 0 ]; then \
	  echo "$@" >> ../Testing/dcoTests/Passed.txt; \
	  echo "$@ passed."; \
	else \
	  echo "$@" >> ../Testing/dcoTests/NotPassed.txt; \
	  echo "$@ failed. Siehe $@.log"; \
	fi