from face_embedding_4 import save_embedding
from load_dataset_2 import save_dataset
from numpy import load
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import pickle
import matplotlib.pyplot as plt

MODEL_PATH = './models/model1.pkl'
TRAIN_PATH = './training_images/'
TEST_PATH = './test_images/test_images/'
EMBEDDING_PATH = './info_model/faces-embeddings.npz'
DATASET_PATH =	'./info_model/dataset.npz'
DICT_PATH = './dataset_dect.json'
# model_location = './models/actual_model.pkl'

def train(dataset_path, embedding_path, model_path, train_path, test_path, dict_path):
    # save dataset
    save_dataset(train_path=train_path, test_path=test_path, save_path=dataset_path, dict_path=dict_path)
    # # save embedding
    save_embedding(save_path=embedding_path, dataset_path=dataset_path)
    # load face embeddings
    data = load(embedding_path)
    #print(data)
    trainX, trainy, testX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
    print(trainX,trainy)
    # normalize input vectors
    in_encoder = Normalizer(norm='l2')
    trainX = in_encoder.transform(trainX)
    testX = in_encoder.transform(testX)
    # label encode targets
    out_encoder = LabelEncoder()
    out_encoder.fit(trainy)
    trainy = out_encoder.transform(trainy)
    # testy = out_encoder.transform(testy)
    # fit model
    # print(trainy)
    model = SVC(kernel='linear', probability=True)
    model.fit(trainX, trainy)
        #for loss
#     plt.figure(1)
#     plt.plot(history.history['loss']) #training loss
#     plt.plot(history.history['val_loss']) #validation loss
#     plt.legend(['training','validation'])
#     plt.title('Loss')
#     plt.xlabel('epoch')

#     #for accuracy
#     plt.figure(2)
#     plt.plot(history.history['accuracy']) #training accuracy
#     plt.plot(history.history['val_accuracy']) #validation accuracy
#     plt.legend(['training','validation'])
#     plt.title('accuracy')
#     plt.xlabel('epoch')
#     plt.show() 

# # Score and Accuracy of our model using unseen test data
#     score = model.evaluate(testX,testy,verbose=0)
#     print('Test Score = ',score[0])
#     print('Test Accuracy = ',score[1])
    # print(history)

    with open(model_path,'wb') as f:
        pickle.dump(model,f)
    print('Training completely sucessfully')
    
if __name__ == "__main__":
    train(DATASET_PATH, EMBEDDING_PATH, MODEL_PATH, TRAIN_PATH, TEST_PATH, DICT_PATH) 

