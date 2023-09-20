.PHONY: help book clean

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  book        to build the book"
	@echo "  clean       to clean out site build files"
	@echo "  publish     to build the book and commit to gh-pages online"

clear:
	find ./content/ -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +

book:
	jupyter-book build ./

publish: book
	ghp-import -n -p -f _build/html

cleanall:
	jupyter-book clean ./ --all

clean:
	jupyter-book clean ./
