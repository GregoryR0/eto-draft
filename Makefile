Image=greg/eto_docker_image

echo:
	echo hello ETO Folks

git-publish:
	#git remote set-url origin git@github.com:tonybutzer/eto-draft.git
	git config --global user.email tonybutzer@gmail.com
	git config --global user.name tonybutzer
	#git config --global user.email skagone@contractor.usgs.gov
	#git config --global user.name skagone
	git config --global push.default simple
	git add .
	git commit 
	# git commit -m "automatic git update from Makefile"
	#python3 ./pkg/ask_commit.py
	git push

build:
	docker build -t ${Image} .

run:
	docker run -it ${Image} bash

exec:
	docker exec -it ${Image} bash

images:
	docker image ls
	echo ================================================================================
	docker image ls | grep greg
