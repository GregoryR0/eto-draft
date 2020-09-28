ls:
	ls ../00-notebooks


nbconvert:
	mkdir -p ./tmp
	echo tmp >> .gitignore
	cp ../00-notebooks/ProjectedReferenceET_Classes_Functions.ipynb tmp/eto_functions.ipynb
	cp ../00-notebooks/ProjectedReferenceET_main_Future+Historical.ipynb tmp/eto_main_api.ipynb
	jupyter nbconvert --to script tmp/eto_functions.ipynb 
	jupyter nbconvert --to script tmp/eto_main_api.ipynb
