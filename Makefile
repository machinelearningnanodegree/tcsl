single_classfier:
	docker build -t timing_comparison .
	docker run -it --rm timing_comparison python app.py CLASSIFIER=$(CLASSIFIER)
	 
all_classifiers:
	docker build -t timing_comparison .
	docker run -it --rm timing_comparison python app.py 
	
notebook_server:
	docker build -t timing_comparison .
	docker run --rm timing_comparison
