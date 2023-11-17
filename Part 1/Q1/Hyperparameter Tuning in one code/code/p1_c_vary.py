
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import  sklearn
from sklearn.model_selection import train_test_split
#import ray
import os
import math
import matplotlib.pyplot as plt
import tensorflow.keras.optimizers.schedules as schedules
import pickle

os.cpu_count()

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

"""# Data processing

For the homework, you should download the dataset with the provided link. The provided dataset has already partitioned the data by clients.
"""

NUM_CLIENT = 100
train_data = 'train_data.npy'
train_data_arr = np.load(train_data, allow_pickle=True)
test_data = 'test_data.npy'
test_data_arr = np.load(test_data, allow_pickle=True)

"""# Define model"""

## adding leanring rate.
def get_model(lr):

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(62, activation="softmax")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

"""## Splitting Data"""

all_train = []
all_val = []
x_train_all = []
y_train_all = []
x_val_all =[]
y_val_all =[]

for i in range(100):

    data = train_data_arr[i]
    # Split the data into training and validation sets
    images_train, images_val, labels_train, labels_val = train_test_split(
        data['images'],
        data['labels'],
        test_size=0.2,
        random_state=42)
    # Normalize the input data
    #images_train = np.array(images_train)
    #images_val = np.array(images_val)
    #images_train, images_val = images_train / 255.0, images_val / 255.0


    # Create TensorFlow datasets
    #expand dimensions of input to (1, 28, 28):
    #print("images before expansion: ", images_train.shape)
    #images_train = np.array([np.expand_dims(image, axis=0) for image in images_train])
    #images_val = [np.expand_dims(image, axis=0) for image in images_val]
    #images_train = np.expand_dims(images_train, axis=1)  # Add batch dimension
    #images_val = np.expand_dims(images_val, axis=1)
    #print("images expanded: ", images_train[0].shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((images_train, labels_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((images_val, labels_val))
    all_train.append(train_dataset)
    all_val.append(val_dataset)

"""# Client class"""

#@ray.remote(num_cpus=1, num_gpus=0)
class Client:
    def __init__(self, i, batch_size, initial_learning_rate, E):
        # Create model
        self.model = get_model(initial_learning_rate)
        self.training_data = all_train[i]
        self.validation_data = all_val[i]
        self.batch_size = batch_size
        self.E = E

    def train(self, global_weights):
        self.model.set_weights(global_weights)


        # Define a custom callback for printing validation metrics every 10 epochs
        '''
        class PrintValidationMetrics(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if epoch % 10 == 0:
                    val_loss, val_accuracy = self.model.evaluate(
                        self.validation_data.batch(self.batch_size), verbose=0)
                    print(f'Epoch {epoch}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

        callback = PrintValidationMetrics()

        '''

        self.model.fit(self.training_data.batch(self.batch_size), epochs=self.E, validation_data=self.validation_data.batch(self.batch_size), verbose=0, shuffle=True)
        ### to get accuracy and loss as arrays for all epochs:
        '''
         history = self.model.fit(self.training_data.batch(self.batch_size), epochs=E, validation_data=self.validation_data.batch(self.batch_size), verbose=0)

        # Access the training history
        ## define objects of these arrays in constructor
        train_loss = history.history['loss']
        train_accuracy = history.history['accuracy']
        val_loss = history.history['val_loss']
        val_accuracy = history.history['val_accuracy']

        return train_loss, train_accuracy, val_loss, val_accuracy

        '''
        return self.model.get_weights()

    def evaluate(self):
        #gives scalars of loss and accuracy at the end of training
        train_loss, train_accuracy = self.model.evaluate(self.training_data.batch(self.batch_size), verbose=0)
        val_loss, val_accuracy = self.model.evaluate(self.validation_data.batch(self.batch_size), verbose=0)
        return train_loss, train_accuracy, val_loss, val_accuracy

"""# Aggregated metrics and losses
Aggregate client metrics and losses for (federated) evaluation metrics.
"""

sample_nbs = []
for i in range(100):
    #print(i)
    #print(len(train_data_arr[i]['labels']))
    #print("train data: ", len( all_train[i]))
    #print("80% of local data: ", len(train_data_arr[i]['labels'])*0.8)
    sample_nbs.append(len(train_data_arr[i]['labels']))
#sample_nbs

def weighted_average(sampled_clients, metric):
    # TODO: Aggregate client losses or accuracies by taking a weighted average.
    #       Weights are proportional to the number of samples at each client
    sample_size_selected = [sample_nbs[i] for i in sampled_clients]
    #sorted list so order should match
    agg_train_acc = sum(weight * acc for weight, acc in zip(sample_size_selected, metric['train_accuracy'])) / sum(sample_size_selected)
    agg_train_loss = sum(weight * loss for weight, loss in zip(sample_size_selected, metric['train_loss'])) / sum(sample_size_selected)
    agg_val_acc = sum(weight * acc for weight, acc in zip(sample_size_selected, metric['val_accuracy'])) / sum(sample_size_selected)
    agg_val_loss = sum(weight * loss for weight, loss in zip(sample_size_selected, metric['val_loss'])) / sum(sample_size_selected)


    return agg_train_acc, agg_train_loss, agg_val_acc, agg_val_loss

"""# Aggregation of client updates"""

def fedAvg(sampled_clients, client_weights):
    # Get the size of samples for each client
    sample_size_selected = [sample_nbs[c] for c in sampled_clients]
    #print("shape of local weights: ", len(client_weights[0]), client_weights[0][0].shape)
    list_dim = len(client_weights[0])
    np_dim = client_weights[0][0].shape
    total = sum(sample_size_selected)

    avg_weights = []
    for k in range(list_dim):
        avg_weights.append(np.zeros(client_weights[0][k].shape))

    for i in range (len(sampled_clients)):
        scalar = sample_size_selected[i]/total
        for j in range(0, list_dim):
            this_numpy = np.multiply(client_weights[i][j], scalar)
            avg_weights[j] = np.add(avg_weights[j],this_numpy)
    #print("shape of avg weights: ", len(avg_weights), avg_weights[0].shape)
    return avg_weights

def plots_train(normalized, t, client, E, batch_size,  train_loss, train_accuracy, val_loss, val_accuracy):
    plt.figure(figsize=(12, 4))
    #plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(normalized + '_train_loss_'+str(t)+'_'+str(client)+'_'+str(E)+'_'+str(batch_size)+'_.png')
    plt.close()
    # Plot training accuracy and validation accuracy
    plt.figure(figsize=(12, 4))
    plt.plot(train_accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(normalized + '_train_accuracy_'+str(t)+'_'+str(client)+'_'+str(E)+'_'+str(batch_size)+'_.png')
    plt.close()

def test(test_data_arr, weights, batch_size, lr):
    model = get_model(lr)  # Construct the model
    model.set_weights(weights)  # Set model weights with the latest parameters

    # Assuming test_data is a tuple of test_images and test_labels
    test_images = np.array(test_data_arr[0]['images'])
    test_labels = np.array(test_data_arr[0]['labels'])



    # Normalize the input data
    #test_images = np.array(test_images)
    #test_images= test_images / 255.0

    test_images_flat = test_images.reshape(test_images.shape[0], -1)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
    batch_size = batch_size  # Set your desired batch size
    # Evaluate the model using the dataset
    batched_test_dataset = test_dataset.batch(batch_size)

    test_loss, test_accuracy = model.evaluate(batched_test_dataset)


    print(f"Test Accuracy: {test_accuracy:.4f}")

    return test_loss, test_accuracy

"""# Launch the learning"""

C_options = [2, 6, 9]
E_options = [300, 60, 10]
LR_options = [0.00001, 0.0001, 0.01]
batch_options = [100, 50, 10]
for C in C_options:
    for E in E_options:
        for initial_learning_rate in LR_options:
            for batch_size  in batch_options:

                run = "C="+str(C)+", E="+str(E)+", LR="+str(initial_learning_rate)+", batch=" + str(batch_size)
                print(run)
                Agg_metric = {}
                Agg_metric['train_loss'] = []
                Agg_metric['train_accuracy'] =[]
                Agg_metric['val_loss'] =[]
                Agg_metric['val_accuracy'] =[]

                global_model = get_model(initial_learning_rate)

                users = np.arange(100)


                total_rounds = 100

                for round in range(total_rounds):
                    # Step 1
                    #print("communication round: ", round)
                    #initiate empty list of model weights
                    client_weights =[]
                    metric = {}
                    metric['train_loss'] = []
                    metric['train_accuracy'] =[]
                    metric['val_loss'] =[]
                    metric['val_accuracy'] =[]
                    # TODO: Sample C fraction of clients
                    sampled_clients = np.random.choice(users, C, replace=False)
                    sampled_clients = sorted(sampled_clients)
                    # TODO: Get the local data for the selected clients
                    for i in sampled_clients:
                        #print("client: ", i)
                        #perform local training
                        #define instance of client:
                        current_client = Client(i, batch_size, initial_learning_rate, E)

                        #save local weights

                        local_weights = current_client.train(global_model.get_weights())
                        #print(local_weights[0].shape)
                        client_weights.append(local_weights)
                        #save local results

                        train_loss, train_accuracy, val_loss, val_accuracy = current_client.evaluate()
                        metric['train_loss'].append(train_loss)
                        metric['train_accuracy'].append(train_accuracy)
                        metric['val_loss'].append(val_loss)
                        metric['val_accuracy'].append(val_accuracy)
                        #print("train_loss : ", train_loss)
                        #print("train_accuracy: ", train_accuracy)


                    # Step 3
                    # TODO: Aggregate the losses and metrics, keep track of the metrics
                    agg_train_acc, agg_train_loss, agg_val_acc, agg_val_loss = weighted_average(sampled_clients, metric)
                    Agg_metric['train_loss'].append(agg_train_loss)
                    Agg_metric['train_accuracy'].append(agg_train_acc)
                    Agg_metric['val_loss'].append(agg_val_loss)
                    Agg_metric['val_accuracy'].append(agg_val_acc)
                    #print("aggregated validation accuracy: ",Agg_metric['val_accuracy'][round])
                    # Step 4
                    # TODO: Aggregate the client updates
                    weighted_avg_weights = fedAvg(sampled_clients, client_weights)

                    # Step 5
                    # Update the global model
                    global_model.set_weights(weighted_avg_weights)

                    if round==99:


                    # Specify the file path
                        file_path = 'Agg_metric_'+str(round)+'_'+run+'_.pickle'

                        # Save the dictionary to a pickle file
                        with open(file_path, 'wb') as file:
                            pickle.dump(Agg_metric, file)

                        #print(f'Dictionary saved to {file_path}')

                        global_model.save('global_model_'+str(round)+'_'+run+'_.h5')
                        #plots_train(run, round, C, E, batch_size,  Agg_metric['train_loss'] , Agg_metric['train_accuracy'], Agg_metric['val_loss'], Agg_metric['val_accuracy'])


                ###
                plots_train(run, total_rounds, C, E, batch_size, Agg_metric['train_loss'] , Agg_metric['train_accuracy'], Agg_metric['val_loss'], Agg_metric['val_accuracy'])
                # Evaluate on test data after training
                
                test_loss, test_accuracy = test(test_data_arr, global_model.get_weights(), batch_size, initial_learning_rate)
                print("train acc: ", Agg_metric['train_accuracy'][99])
                print("validation acc: ", Agg_metric['val_accuracy'][99])
                print("test loss: ", test_loss)
                print("test accuracy: ", test_accuracy)


                test_results = np.array ([test_loss, test_accuracy])
                np.save('test_results_'+run+'_'+str(total_rounds)+'_.npy', test_results)













