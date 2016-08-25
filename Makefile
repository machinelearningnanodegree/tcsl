wrangle_data:
	docker run -it -v $(shell pwd):/home/jovyan/work \
	  --rm jupyter/scipy-notebook python -m lib.data.wrangler

single_classfier:
	docker run -it -v $(shell pwd):/home/jovyan/work \
	  --rm jupyter/scipy-notebook python app.py CLASSIFIER=$(CLASSIFIER)

all_classifiers:
	docker run -it -v $(shell pwd):/home/jovyan/work \
	  --rm jupyter/scipy-notebook python app.py

notebook_server:
	docker run -v $(shell pwd):/home/jovyan/work \
	  --rm jupyter/scipy-notebook

clean:
	rm -rf tmp results **/*.pyc	**/__pycache__
