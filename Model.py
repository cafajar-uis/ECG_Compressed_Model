import numpy as np 
import numpy.matlib
import os 
import tempfile

import tensorflow as tf

tf.compat.v1.enable_eager_execution()
tf.executing_eagerly()

import tensorflow.keras as keras
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers 
from tensorflow.keras import layers
from sklearn.model_selection import StratifiedKFold


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

GPU1 = "6"
os.environ["CUDA_DEVICE_ORDER"]    ="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"] = GPU1 


If_test  = 3     # 0 es prueba corta, 1 es prueba media-baja, 2 es prueba media-alta y 3 es una prueba completa
If_kfold = True  # True es entrenar con kfold, False es sin kfold
If_train = False  # True es entrenar, False es cargar modelo ya entrenado
If_prune = True  # True es prunear, False es cargar modelo ya pruned
If_save  = False  # True es guardar modelo, False es no guardar


def load_data(): # Full data
    X_train_r = np.loadtxt('../../Data/Data_New_3l.dat', delimiter=',')
    X_train_r = np.array(X_train_r, dtype='float32').reshape(54593,1500,1)
    y_train = np.loadtxt('../../Data/Labels_New_3l.dat', delimiter=',')
    y_train = np.array(y_train, dtype='float32').reshape(54593,1)
    return X_train_r, y_train    


def load_data_toy(): # Toy data
    X_train_r = np.loadtxt('../../Data/Data_New_3l_toy.dat', delimiter=',')
    X_train_r = np.array(X_train_r, dtype='float32').reshape(3108,1500,1)
    y_train = np.loadtxt('../../Data/Labels_New_3l_toy.dat', delimiter=',')
    y_train = np.array(y_train, dtype='float32').reshape(3108,1)
    return X_train_r, y_train    

    
print('[Info]: Loading dataset...')
if If_test == 0 or If_test == 1:
    X_train, y_train = load_data_toy()
    print('[Info]: Toy Dataset Load')
else:
    X_train, y_train = load_data()
    print('[Info]: Full Dataset Load')
 

def zeropad(x, fils):  # Concatena zeros para igualar dimensiones
        pad1 = K.zeros_like(x)
        assert (fils % pad1.shape[2]) == 0
        num_repeat = fils // pad1.shape[2]
        for i in range(num_repeat - 1):
            x = K.concatenate([x, pad1], axis=2)
        return x 
    
    
def basic_block(x_in, pool_size, strides, filters, kernel_size, DP):
    x = layers.BatchNormalization(axis=-1)(x_in)
    y = layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding='same')(x_in)
    y = layers.Lambda(zeropad, arguments={'fils':filters})(y) 
    x = layers.ReLU()(x) 
    x = layers.Conv1D(filters = filters, kernel_size = kernel_size ,padding='same')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DP)(x)
    x = layers.Conv1D(filters = filters, kernel_size = kernel_size, padding='same')(x)
    x = layers.AveragePooling1D(pool_size=pool_size, strides=strides, padding='same')(x)
    x = layers.Add()([y,x])
    return x


# % [Step]: Training the model

# Empty lists for save the results
Filters_initial_fold = []
F1_per_fold = []
Acc_Val_per_fold = []
Acc_total = []
F1_total = []


if If_test == 0:           #Toy test short
    Total_epochs = 3
    N_cnn = 5
    num_folds = 3
    filters_initial = 8
    N_kernel = 16
    DP1=0.2
    DP2=0.2
elif If_test == 1:         #Toy test long
    Total_epochs = 15
    N_cnn = 13
    num_folds = 5
    filters_initial = 32
    N_kernel = 16
    DP1=0.2
    DP2=0.2
elif If_test == 2:         #Full test short
    Total_epochs = 10
    num_folds = 3
    N_cnn = 13
    filters_initial = 32
    N_kernel = 16
    DP1=0.2
    DP2=0.2
else:                      #Full test long
    Total_epochs = 100
    num_folds = 5
    N_cnn = 13
    filters_initial = 32
    N_kernel = 16
    DP1=0.2
    DP2=0.2
    
    
# Training parameters
learning_rate = 0.01
adam = optimizers.Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
skf = StratifiedKFold(n_splits=num_folds, shuffle=True)


num_cnn = N_cnn   
kernel_size = N_kernel
num_classes = 3
input_shape = (X_train.shape[1], 1)
fold_no = 1
DP=0.2
batch_size = 128
pool_size=2
strides=2
k=0

####################################################################
##################### MODEL #######################################
 
filters = filters_initial*(2**k) # Modificar el numero de salidas de la capas convoluciones
input_signal = keras.Input(shape=input_shape, name='img')
x = layers.Conv1D(filters = filters, kernel_size = kernel_size, padding='same')(input_signal)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.ReLU()(x)
y = layers.MaxPooling1D(pool_size=pool_size, strides=strides, padding='same')(x)
y = layers.Lambda(zeropad, arguments={'fils':filters})(y) 
x = layers.Conv1D(filters = filters, kernel_size = kernel_size, padding='same')(x)
x = layers.BatchNormalization(axis=-1)(x)
x = layers.ReLU()(x)
x = layers.Dropout(DP)(x)
x = layers.Conv1D(filters = filters, kernel_size = kernel_size, strides=strides, padding='same')(x)
x = layers.Add()([y,x])

for i in range(num_cnn):
     if i%8 == 0:
         filters = filters_initial*(2**k)
         k = k + 1 
         strides = 2
         DP = DP1
         x = basic_block(x, pool_size, strides, filters, kernel_size, DP)
     else:
         strides = 1 
         DP= DP2   
         x = basic_block(x, pool_size, strides, filters, kernel_size, DP)
         
x = layers.BatchNormalization(axis=-1)(x)        
x = layers.ReLU()(x)
x = layers.Flatten()(x)
outputs = layers.Dense(3)(x)  # Sin softmax para generar los logits para el KD

####################################################################

model = keras.Model(inputs=input_signal, outputs=outputs, name='ecg_model')
model.compile(loss=keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer='adam',
              metrics=['accuracy'],
              run_eagerly=True)         # Para poder cuantizar
model.summary()
num_layers = len(model.layers)-1
model.save('models/Initial_model.h5')

#%% Entrenar con kfold

if If_kfold == True and If_train == True:
    num_classes = 3
    inputs = X_train
    targets = y_train
    fold_no = 1
    
    for train, test in skf.split(inputs, targets): 
        
        print('\n[Info]: Traning for the fold ', fold_no)
        
        model_path = 'models/Best_Model'+str(fold_no)+'.h5'
        callbacks_list = [
        
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            mode='auto',
            min_delta=1e-3,
            patience=5,
            verbose=1,
            restore_best_weights=True),  
        
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='auto',
            factor=0.1,
            patience=3,
            min_lr=0.00000000001),
        
        
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor='val_loss',
            mode='auto',
            verbose=1,
            save_best_only=True)
        ]
                
        if fold_no == 1:
            targets = keras.utils.to_categorical(targets-1, num_classes)
            
        X_mean = np.mean(inputs[train])
        X_std = np.std(inputs[train])
        X_train_fold = (inputs[train] - X_mean)/X_std
        X_test_fold  = (inputs[test] - X_mean)/X_std
        Y_train_fold = targets[train]
        Y_test_fold  = targets[test]
        
        model.load_weights('models/Initial_model.h5') 
        K.set_value(model.optimizer.learning_rate, 0.01)
        
        history = model.fit(X_train_fold, Y_train_fold, 
                                      epochs=Total_epochs, 
                                      batch_size=128,
                                      callbacks=callbacks_list,
                                      validation_split=0.1)
        
       
        print('[Info]: Loading the Best Model ...')
        model.load_weights(model_path)
        print('[Info]: Best Model load!')
        (loss, acc_full) = model.evaluate(X_test_fold, Y_test_fold, verbose=1)
        print(f'[Info]: Val_accu = {acc_full:.4} %')
        Acc_Val_per_fold.append(acc_full)
           
        
        # Calculating the F1
        y_pred_for_f1 = model.predict(X_test_fold)
        y_true_for_f1 = np.argmax(Y_test_fold, axis = -1)
        y_pred_for_f1_2 = np.argmax(y_pred_for_f1, axis=-1)
        CF = confusion_matrix(y_true_for_f1,y_pred_for_f1_2)
        P2 = CF.sum(axis=0)
        R2 = CF.sum(axis=1)
        D2 = R2+P2
        F1i_2 = 2*np.diag(CF)/D2
        F1_med_2 = F1i_2[0:2]
        F1_2 = F1_med_2.sum(axis=0)/2
        print(f'[Info]: F1 = {F1_2:.8} %')
        F1_per_fold.append(F1_2)
        
        
        print('------------------------------------------------------------------------ ')
        print('[Info]: Updating Results_training_base.txt file')
        with open('Results_training_base.txt', 'w+') as f: 
                f.write(" ".join(map(str, F1_per_fold)))
                f.write("\n" )
                f.write(" ".join(map(str, Acc_Val_per_fold)))
                f.write("\n" )
                f.close()
    
        fold_no = fold_no + 1
    
    if If_save == True:     
      model.save('models/Trained_full_kfolded_model.h5')
      model.save_weights('models/Trained_full_kfolded_weights.h5')

#%% Cargar modelos entrenados

if If_train == False:
    if If_kfold == True:
        num_classes = 3
        inputs = X_train
        targets = y_train
        fold_no = 1
        acc_temp = []
        F1_temp = []
        for train, test in skf.split(inputs, targets):
        
            print('\n[Info]: Evaluating for the fold ', fold_no)
            
            if fold_no == 1:
                targets = keras.utils.to_categorical(targets-1, num_classes)   
                
            X_mean = np.mean(inputs[train])
            X_std = np.std(inputs[train])
            X_train_fold = (inputs[train] - X_mean)/X_std
            X_test_fold = (inputs[test] - X_mean)/X_std
            Y_train_fold = targets[train]
            Y_test_fold  = targets[test]
            
            model_path = 'models/Best_Model'+str(fold_no)+'.h5'
            model.load_weights(model_path)
            (loss, acc) = model.evaluate(X_test_fold, Y_test_fold, verbose=0)
            acc_temp.append(acc)
            print(f'acc = {acc:.8} %')
            
            # Calculating the F1
            y_pred_for_f1 = model.predict(X_test_fold)
            y_true_for_f1 = np.argmax(Y_test_fold, axis = -1)
            y_pred_for_f1_2 = np.argmax(y_pred_for_f1, axis=-1)
            CF = confusion_matrix(y_true_for_f1,y_pred_for_f1_2)
            P2 = CF.sum(axis=0)
            R2 = CF.sum(axis=1)
            D2 = R2+P2
            F1i_2 = 2*np.diag(CF)/D2
            F1_med_2 = F1i_2[0:2]
            F1_2 = F1_med_2.sum(axis=0)/2
            print(f'F1 = {F1_2:.8} %')
            F1_temp.append(F1_2)
            
            print('------------------------------------------------------------------------ ')
            print('[Info]: Updating Results_inference_base.txt file')
            with open('Results_inference_base.txt', 'w+') as f: 
                f.write(" ".join(map(str, F1_temp)))
                f.write("\n" )
                f.write(" ".join(map(str, acc_temp)))
                f.write("\n" )
                f.close()
            
            fold_no = fold_no + 1
            
            
    else:
        model = keras.models.load_model('models/Trained_full_wo_kfolded_model.h5')
        inputs = X_train
        targets = y_train
        X_train, X_test, y_train, y_test = train_test_split(inputs, targets, test_size=0.2)
        y_train_cat = keras.utils.to_categorical(y_train-1, 3)
        y_test_cat  = keras.utils.to_categorical(y_test-1, 3)
        
#%% Evaluar
print()
print("Original metrics average:")
acc_original  = sum(acc_temp)/len(acc_temp)
F1_original   = sum(F1_temp)/len(F1_temp)
print(f'Accuracy_original = {acc_original:.4} %')
print(f'F1_original = {F1_original:.4} %')

#%% Clone model
print()
print('Cloning Model...')
student = keras.models.clone_model(model)
student.set_weights(model.get_weights())
student.compile(optimizer='adam',
                loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'],
                run_eagerly=True)
                
############################################ PRUNING ########################################

import tensorflow_model_optimization as tfmot

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# Compute end step to finish pruning after 2 epochs.
batch_size = 128
epochs = 100
validation_split = 0.1 # 10% of training set will be used for validation set. 
constant_sparsity = 0.80

num_images = X_train_fold.shape[0] * (1 - validation_split)
end_step = np.ceil(num_images / batch_size).astype(np.int32) * epochs

pruning_params = {
      'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(constant_sparsity,
                                                                begin_step=0,
                                                                end_step=-1)
}

student_pruned = prune_low_magnitude(student, **pruning_params)

student_pruned.compile(optimizer='adam',
                       loss=keras.losses.CategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'],
                       run_eagerly=True)
student_pruned.save('models/Initial_pruned_model.h5')

logdir = tempfile.mkdtemp()

num_classes = 3
inputs = X_train
targets = y_train
fold_no = 1

F1_per_fold_pruned = []
Acc_Val_per_fold_pruned = []
Acc_total_pruned = []
F1_total_pruned = []

if If_prune == True:
  for train, test in skf.split(inputs, targets):
      print('\n[Info]: Traning pruning for the fold ', fold_no)
      
      model_path_student_pruned = 'models/Best_Model_Pruning'+str(fold_no)+'.h5'
      
      callbacks_student_pruned = [
      keras.callbacks.EarlyStopping(
          monitor='val_loss',
          mode='auto',
          min_delta=1e-3,
          patience=5,
          verbose=1,
          restore_best_weights=True),  
      keras.callbacks.ReduceLROnPlateau(
          monitor='val_loss',
          mode='auto',
          factor=0.1,
          patience=4,
          min_lr=0.00000000001),
      keras.callbacks.ModelCheckpoint(
          filepath=model_path_student_pruned,
          save_weights_only=True,
          monitor='val_loss',
          mode='auto',
          save_best_only=True),
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir)
      ]
              
      if fold_no == 1:
          targets = keras.utils.to_categorical(targets-1, num_classes)
          
      X_mean = np.mean(inputs[train])
      X_std = np.std(inputs[train])
      X_train_fold = (inputs[train] - X_mean)/X_std
      X_test_fold  = (inputs[test] - X_mean)/X_std   
      Y_train_fold = targets[train]
      Y_test_fold  = targets[test]
      
      student_pruned.load_weights('models/Initial_pruned_model.h5') 
      K.set_value(student_pruned.optimizer.learning_rate, 0.01)
      
      history_student_pruned = student_pruned.fit(X_train_fold, Y_train_fold, 
                                                  epochs=epochs, 
                                                  batch_size=batch_size,
                                                  validation_split=validation_split,
                                                  callbacks=callbacks_student_pruned)
      
      
      print('[Info]: Loading the Best Model ...')
      student_pruned.load_weights(model_path_student_pruned)
      print('[Info]: Best Model load!')
      (loss, acc_full) = student_pruned.evaluate(X_test_fold, Y_test_fold, verbose=1)
      print(f'[Info]: Val_accu = {acc_full:.4} %')
      Acc_Val_per_fold_pruned.append(acc_full)
         
      
      # Calculating the F1
      y_pred_for_f1 = student_pruned.predict(X_test_fold)
      y_true_for_f1 = np.argmax(Y_test_fold, axis = -1)
      y_pred_for_f1_2 = np.argmax(y_pred_for_f1, axis=-1)
      CF = confusion_matrix(y_true_for_f1,y_pred_for_f1_2)
      P2 = CF.sum(axis=0)
      R2 = CF.sum(axis=1)
      D2 = R2+P2
      F1i_2 = 2*np.diag(CF)/D2
      F1_med_2 = F1i_2[0:2]
      F1_2 = F1_med_2.sum(axis=0)/2
      print(f'[Info]: F1 = {F1_2:.8} %')
      F1_per_fold_pruned.append(F1_2)
      
      
      print('------------------------------------------------------------------------ ')
      print('[Info]: Updating Results_training_pruned.txt file')
      with open('Results_training_pruned.txt', 'w+') as f: 
              f.write(" ".join(map(str, F1_per_fold_pruned)))
              f.write("\n" )
              f.write(" ".join(map(str, Acc_Val_per_fold_pruned)))
              f.write("\n" )
              f.close()
      
      fold_no = fold_no + 1
      
  student_pruned.save('models/Trained_full_kfolded_pruned_model.h5')
  student_pruned.save_weights('models/Trained_full_kfolded_pruned_weights.h5')
  
else:
  acc_temp_pruned = []
  F1_temp_pruned = []
  for train, test in skf.split(inputs, targets):
    print('\n[Info]: Evaluating pruning for the fold ', fold_no)
             
    if fold_no == 1:
        targets = keras.utils.to_categorical(targets-1, num_classes)
        
    X_mean = np.mean(inputs[train])
    X_std  = np.std(inputs[train])  
    X_train_fold = (inputs[train] - X_mean)/X_std
    X_test_fold  = (inputs[test] - X_mean)/X_std
    Y_train_fold = targets[train]
    Y_test_fold  = targets[test]

    model_path_student_pruned = 'models/Best_Model_Pruning'+str(fold_no)+'.h5'
    student_pruned.load_weights(model_path_student_pruned)
    (loss, acc) = student_pruned.evaluate(X_test_fold, Y_test_fold, verbose=1)
    acc_temp_pruned.append(acc)
    print(f'acc = {acc:.8} %')
       
    
    # Calculating the F1
    y_pred_for_f1 = student_pruned.predict(X_test_fold)
    y_true_for_f1 = np.argmax(Y_test_fold, axis = -1)
    y_pred_for_f1_2 = np.argmax(y_pred_for_f1, axis=-1)
    CF = confusion_matrix(y_true_for_f1,y_pred_for_f1_2)
    P2 = CF.sum(axis=0)
    R2 = CF.sum(axis=1)
    D2 = R2+P2
    F1i_2 = 2*np.diag(CF)/D2
    F1_med_2 = F1i_2[0:2]
    F1_2 = F1_med_2.sum(axis=0)/2
    print(f'F1 = {F1_2:.8} %')
    F1_temp_pruned.append(F1_2)
    
    print('------------------------------------------------------------------------ ')
    print('[Info]: Updating Results_inference_pruned.txt file')
    with open('Results_inference_pruned.txt', 'w+') as f: 
            f.write(" ".join(map(str, F1_temp_pruned)))
            f.write("\n" )
            f.write(" ".join(map(str, acc_temp_pruned)))
            f.write("\n" )
            f.close()
    
    fold_no = fold_no + 1
    
#%% Evaluar
print()
print("Pruned metrics average:")
acc_pruned = sum(acc_temp_pruned)/len(acc_temp_pruned)
F1_pruned  = sum(F1_temp_pruned)/len(F1_temp_pruned)
print(f'Accuracy_pruned = {acc_pruned:.4} %')
print(f'F1_pruned = {F1_pruned:.4} %')

#%% Removing wrappers
student_pruned_stripped = tfmot.sparsity.keras.strip_pruning(student_pruned)
student_pruned_stripped.save('models/student_pruned_stripped.h5')

#%% Funciones para leer tamaño y parametros
import tempfile
def get_gzipped_model_size(model):
  # Returns size of gzipped model, in bytes.
  import os
  import zipfile

  _, keras_file = tempfile.mkstemp('.h5')
  model.save(keras_file, include_optimizer=False)

  _, zipped_file = tempfile.mkstemp('.zip')
  with zipfile.ZipFile(zipped_file, 'w', compression=zipfile.ZIP_DEFLATED) as f:
    f.write(keras_file)

  return os.path.getsize(zipped_file)

def get_params_nonzero(model):
    params = 0
    for layer in model.layers:
        for weight in layer.get_weights():
            params += np.count_nonzero(weight.flatten())
    return params
#%%
print()
print('SUMMARY:')
print()
print('Sparisty:', constant_sparsity)
print()
print('Baseline average accuracy: %.4f' % acc_original) 
print('Pruned average accuracy:   %.4f' % acc_pruned)
print('Baseline average F1: %.4f' % F1_original) 
print('Pruned average F1:   %.4f' % F1_pruned)
print()
print('Parameters of baseline model:       ' , f'{(get_params_nonzero(model)):,}'  )
print('Parameters of pruned model:         ' , f'{(get_params_nonzero(student_pruned)):,}'   )
print('Parameters of pruned stripped model:' , f'{(get_params_nonzero(student_pruned_stripped)):,}'   )
print('Parameters ratio: %.2f' % ((get_params_nonzero(model))/get_params_nonzero(student_pruned_stripped)) )
print()
print('Size of baseline model:       ' , f'{(get_gzipped_model_size(model)):,}' , 'Bytes')
print('Size of pruned model:         ' , f'{(get_gzipped_model_size(student_pruned)):,}' , 'Bytes')
print('Size of pruned stripped model:' , f'{(get_gzipped_model_size(student_pruned_stripped)):,}' , 'Bytes')
print()
print('Compression pruned model:          %.2f' % ( (get_gzipped_model_size(model))/(get_gzipped_model_size(student_pruned))) )
print('Compression pruned stripped model: %.2f' % ( (get_gzipped_model_size(model))/(get_gzipped_model_size(student_pruned_stripped))) )


