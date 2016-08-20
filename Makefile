wrangle_data:
	docker build -t timing_comparison .
	docker run -it -v $(shell pwd):/home/jovyan/work --rm timing_comparison python -m lib.data.wrangler

single_classfier:
	docker build -t timing_comparison .
	docker run -it --rm timing_comparison python app.py CLASSIFIER=$(CLASSIFIER)

all_classifiers:
	docker build -t timing_comparison .
	docker run -it --rm timing_comparison python app.py

notebook_server:
	docker build -t timing_comparison .
	docker run --rm timing_comparison

clean:
	rm -rf tmp results **/*.pyc	**/__pycache__
