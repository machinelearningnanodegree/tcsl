import time
from sklearn.metrics import f1_score

class Model:
    def __init__(self, classifier, parameters={}):
        self.classifier = classifier(**parameters)
        self.training_time = None
        self.length_training_set = None
        self.f1_train, self.train_prediction_time = None, None
        self.f1_test, self.test_prediction_time = None, None

    def train_classifier(self, X_train, y_train):
        print("Training {}...".format(self.classifier.__class__.__name__))
         # start = np.datetime64(datetime.datetime.now(),"us")
        start = time.time()
        self.classifier.fit(X_train, y_train)
        # end = np.datetime64(datetime.datetime.now(),"us")
        end = time.time()
        self.training_time = end - start

    def predict_labels(self, features, target):
        # print("Predicting labels using {}...".format(self.classifier.__class__.__name__))
         # start = np.datetime64(datetime.datetime.now(),"us")
        start = time.time()
        y_pred = self.classifier.predict(features)
        # end = np.datetime64(datetime.datetime.now(),"us")
        end = time.time()
        prediction_time = end - start
        f1_score_output = f1_score(target, y_pred, average="macro")
        return f1_score_output, prediction_time

    def __str__(self):
        return """
Model(classifer: {}
      length training set: {}
      training time: {}
      train f1/pred. time: {} {}
      test f1/pred. time: {} {})
               """.format(self.classifier.__class__.__name__,
                          self.length_training_set,
                          self.training_time, self.f1_train, self.train_prediction_time,
                          self.f1_test, self.test_prediction_time)

    def __repr__(self):
        return """
Model(classifer: {}
      length training set: {}
      training time: {}
      train f1/pred. time: {} {}
      test f1/pred. time: {} {})

Detailed Classifier Description:
{}
               """.format(self.classifier.__class__.__name__,
                          self.length_training_set,
                          self.training_time, self.f1_train, self.train_prediction_time,
                          self.f1_test, self.test_prediction_time, self.classifier)

    def __call__(self, X_train, y_train, X_test, y_test):
        self.length_training_set = len(X_train)
        self.training_time = self.train_classifier(X_train, y_train)
        self.f1_train, self.train_prediction_time = self.predict_labels(X_train, y_train)
        self.f1_test, self.test_prediction_time = self.predict_labels(X_test, y_test)
