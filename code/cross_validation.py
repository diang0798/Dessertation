from core import Trainer_experiment1
from core import Trainer_experiment3
from core import Trainer_experiment4
from core import Trainer_experiment5
from utils import index_loader
from core import data_test

# Parameter Settings
num_epochs = 800
learning_rate = 0.0001
batch_size = 5
norm = 32
current_experiment = 5

# Cross Validation
for i in range(3):

    training_set, validation_set = index_loader.dataset_generator(norm=norm, experiment=current_experiment,
                                                                  train_batch_size=batch_size, cr_round=i)

    if current_experiment == 1 or current_experiment == 2:
        Trainer_experiment1.experiment1(num_epochs=num_epochs, training_loader=training_set,
                                        validation_loader=validation_set, lr=learning_rate, cr_round=i,
                                        experiment=current_experiment)

    if current_experiment == 3:
        Trainer_experiment3.experiment3(num_epochs=num_epochs, training_loader=training_set,
                                        validation_loader=validation_set, lr=learning_rate, cr_round=i,
                                        batch_size=batch_size, norm=norm)

    if current_experiment == 4:
        Trainer_experiment4.experiment4(num_epochs=num_epochs, training_loader=training_set,
                                        validation_loader=validation_set, lr=learning_rate, cr_round=i,
                                        batch_size=batch_size, norm=norm)

    if current_experiment == 5:
        Trainer_experiment5.experiment5(num_epochs=num_epochs, training_loader=training_set,
                                        validation_loader=validation_set, lr=learning_rate, cr_round=i,
                                        batch_size=batch_size)

    data_test.testing(experiment=current_experiment, cr_round=i, norm=norm)


