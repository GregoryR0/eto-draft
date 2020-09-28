# eto github repo

## Goals

1. Package the reference et python code into a pip installable package
2. Store all code in source code control and publish in git
3. Allow for refactoring of code into API and LIB
4. Build an rouze/eto docker image and run the darn thing!

## Running the Code

- getting used to the terminal
- cd api_eto
	- python3 eto --help
	- python3 eto ---

## Building the Library

- cd etoLib
- make build

## Self Documeting Approaches
- Makefiles
- Dockerfile

## Steps

### Todo

- create etoLib directory path and stub out functions and classes
- setup.py in etoLib
- refactor - see refactor code section
- create api_eto for main line stubs 



### Completed

- create a greg.ipynb for viewing markdown such as this Readme.md file
- create directory for notional git repo - done
	- ~/00-eto-github-prep
- convert ipynb into .py script files
	- see ~/shell/Makefile for how this was done
- fix vim colors for dark background
	- set background=dark
