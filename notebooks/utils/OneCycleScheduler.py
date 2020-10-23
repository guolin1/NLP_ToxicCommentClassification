import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K

class OneCycleScheduler(Callback):
    """This callback implements a cyclic scheduler for learning rate 
    and momentum (for SGD and RMSProp; beta_1 for Adam).
    The method cycles through learning rate and momentum with constant 
    frequency (step size).
    
    # Arguments
        max_lr: upper boundary in the learning rate cycle. 
        base_lr: start of the cycle. If undefined, this is set 
            to be 25x smaller than the upper boundary. Note that 
            the mininum learning rate (the end of the cycle) is 
            100x smaller than the base_lr as recommended. 
        initial_mom: the start of the momentum cycle, it's also the peak
        min_mom: the minimum of the momentum cycle. 
        step_size: number of training iterations per
            half cycle. 
            
    # Returns
        history: A dictionary that contains the learning rates and momentums
            used as a function of steps (batch number). See usage below.
    
    For more detail, please see paper (referenced below).

    # Example for CIFAR10 w/ batch size 100 (Note that you may want to try
      to use a stepsize that's 2-8x the number of batches):
        ```python
              bs = 100
              steps_per_epoch = len(x_train)/bs
              lr_scheduler = OneCycleScheduler(
                                        max_lr=5e-2, 
                                        step_size = steps_per_epoch/2)
              model.fit(x_train, 
                        y_train, 
                        batch_size=bs, 
                        steps_per_epoch=steps_per_epoch,
                        epochs=4,
                        callbacks=[lr_scheduler])
        ```

    # References
      Smith, Leslie.N. (2018) A disciplined approach to neural network hyper-
          parameters: Part 1 -- learning rate, batch size, momentum, and 
          weight decay. (https://arxiv.org/abs/1803.09820).
    """

    def __init__(
            self,
            max_lr=0.6,
            base_lr=None,
            base_mom = 0.9,
            min_mom = 0.9,
            step_size=10.,
            mode = 'triangular'):
      
        super(OneCycleScheduler, self).__init__()

        if mode not in ['triangular', 'triangular2']:
            raise KeyError("mode must be one of 'triangular', "
                           "or 'triangular2'")
        self.max_lr = max_lr
        self.base_lr = base_lr
        self.base_mom = base_mom
        self.min_mom = min_mom
        self.step_size = step_size
        self.mode = mode
        self.max_lr_tracker = max_lr
        self.i_iteration = 0.
        self.history = {}

    def clr_mom(self):
        '''
        Compute learning rate for a given step
        '''
        ## Get current cycle number
        cycle = np.floor(1 + self.i_iteration / (2 * self.step_size))
        ## Half the peak lr at each cycle if 'triangular2' is selected.
        if ((self.mode == 'triangular2') &
            (self.max_lr_tracker != np.clip(self.max_lr / (2**cycle-1), self.base_lr, self.max_lr))):
            new_max = np.clip(self.max_lr / (2**cycle-1), self.base_lr, self.max_lr)
            lin_lrs = np.linspace(self.base_lr,new_max,int(self.step_size*0.9)) # raise to max 
            lin_lrs_2 = np.linspace(new_max, self.base_lr, int(self.step_size*0.9)) # drop down to min
            lin_lrs_3 = np.linspace(self.base_lr, self.base_lr/100, int(self.step_size*0.2)) # dip further down
            self.cycle_lrs = np.concatenate((lin_lrs,lin_lrs_2,lin_lrs_3)) # concatenate 3 segments 
            self.max_lr_tracker = new_max
        
        ## Get where the current iteration is (in %) within the cycle
        whereami = (2*self.step_size - (cycle*(2*self.step_size)-(self.i_iteration+1)))/(2*self.step_size)
        ## Grab lr and mom from predefined cyclic learning rates and moms
        return (self.cycle_lrs[np.clip(int(whereami*(self.step_size*2)),0,len(self.cycle_lrs)-1)], 
                self.cycle_moms[np.clip(int(whereami*(self.step_size*2)),0,len(self.cycle_moms)-1)])

    def on_train_begin(self, logs=None):
        '''
        At the beginning of training we initialize the learning rate and
        momentum starting base learning rate (base_lr) and maximum momentum 
        (mom_max). We also create the cyclic learning rates and momentum
        following Leslie N Smith's 2018 paper https://arxiv.org/abs/1803.09820. 
        '''
        logs = logs or {}
        if self.base_lr is None: self.base_lr = self.max_lr/25
        if self.i_iteration == 0:
            ## initialize learning rate and momentum
            K.set_value(self.model.optimizer.lr, self.base_lr)
            if hasattr(self.model.optimizer, 'momentum'):
                K.set_value(self.model.optimizer.momentum, self.base_mom)
            else:
                K.set_value(self.model.optimizer.beta_1, self.base_mom)
            ## Create cyclic learning rates
            lin_lrs = np.linspace(self.base_lr,self.max_lr,int(self.step_size*0.9)) # raise to max 
            lin_lrs_2 = np.linspace(self.max_lr, self.base_lr, int(self.step_size*0.9)) # drop down to min
            lin_lrs_3 = np.linspace(self.base_lr, self.base_lr/100, int(self.step_size*0.2)) # dip further down
            self.cycle_lrs = np.concatenate((lin_lrs,lin_lrs_2,lin_lrs_3)) # concatenate 3 segments 
            ## Create cyclic momentums
            lin_moms = np.linspace(self.base_mom, self.min_mom,int(self.step_size*0.9)) # drop to minimum 
            lin_moms_2 = np.linspace(self.min_mom, self.base_mom, int(self.step_size*0.9)) # raise to base (max)
            lin_moms_3 = np.linspace(self.base_mom, self.base_mom, int(self.step_size*0.2)) # remain at base
            self.cycle_moms = np.concatenate((lin_moms,lin_moms_2,lin_moms_3)) # concatenate 3 segments 
        else:
            ## if lr already initialized, call clr_mom() to update lr and mom.
            ilr, imom = self.clr_mom()
            K.set_value(self.model.optimizer.lr, ilr)
            if hasattr(self.model.optimizer, 'momentum'):
                K.set_value(self.model.optimizer.momentum, imom())
            else:
                K.set_value(self.model.optimizer.beta_1, imom())
            
    def on_batch_end(self, epoch, logs=None):
        '''
        At the end of each batch, we increase the iteration number to reflect
        the next step, and then compute the learning rate and momentum for that 
        step. We also save the learning rate and momentum into a dict.
        '''
        logs = logs or {}
        ## save lrs and moms in history
        if 'lr' in self.history:
            self.history['lr'].append(self.model.optimizer.lr.numpy())
            if hasattr(self.model.optimizer, 'momentum'):
                self.history['mom'].append(self.model.optimizer.momentum.numpy())
            else:
                self.history['mom'].append(self.model.optimizer.beta_1.numpy())
        else:
            self.history.setdefault('lr',[])
            self.history.setdefault('mom',[])
            self.history['lr'].append(self.model.optimizer.lr.numpy())
            if hasattr(self.model.optimizer, 'momentum'):
                self.history['mom'].append(self.model.optimizer.momentum.numpy())
            else:
                self.history['mom'].append(self.model.optimizer.beta_1.numpy())
        ## Update lr and mom
        self.i_iteration += 1 # track iteration number
        ilr, imom = self.clr_mom() # compute lr and mom
        K.set_value(self.model.optimizer.lr, ilr) # set lr
        if hasattr(self.model.optimizer, 'momentum'): # set mom
            K.set_value(self.model.optimizer.momentum, imom)
        else:
            K.set_value(self.model.optimizer.beta_1, imom)

class LRFinder(Callback):
    """
    Credit: https://github.com/avanwyk/tensorflow-projects/tree/master/lr-finder

    Callback that exponentially adjusts the learning rate after each training batch between start_lr and
    end_lr for a maximum number of batches: max_step. The loss and learning rate are recorded at each step allowing
    visually finding a good learning rate as per https://sgugger.github.io/how-do-you-find-a-good-learning-rate.html via
    the plot method.
    """

    def __init__(self, start_lr: float = 1e-7, end_lr: float = 10, max_steps: int = 100, smoothing=0.9):
        super(LRFinder, self).__init__()
        self.start_lr, self.end_lr = start_lr, end_lr
        self.max_steps = max_steps
        self.smoothing = smoothing
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_begin(self, logs=None):
        self.step, self.best_loss, self.avg_loss, self.lr = 0, 0, 0, 0
        self.lrs, self.losses = [], []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr = self.exp_annealing(self.step)
        tf.keras.backend.set_value(self.model.optimizer.lr, self.lr)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        step = self.step
        if loss:
            self.avg_loss = self.smoothing * self.avg_loss + (1 - self.smoothing) * loss
            smooth_loss = self.avg_loss / (1 - self.smoothing ** (self.step + 1))
            self.losses.append(smooth_loss)
            self.lrs.append(self.lr)

            if step == 0 or loss < self.best_loss:
                self.best_loss = loss

            if smooth_loss > 4 * self.best_loss or tf.math.is_nan(smooth_loss):
                self.model.stop_training = True

        if step == self.max_steps:
            self.model.stop_training = True

        self.step += 1

    def exp_annealing(self, step):
        return self.start_lr * (self.end_lr / self.start_lr) ** (step * 1. / self.max_steps)

    def plot(self):
        '''
        plot losses as a function of lrs
        '''
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs, self.losses)
        return fig, ax
    
    def plot_changes(self):
        '''
        plot changes in loss as a function of lrs
        '''
        fig, ax = plt.subplots(1, 1)
        ax.set_ylabel('Change in Loss [%]')
        ax.set_xlabel('Learning Rate (log scale)')
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
        ax.plot(self.lrs[1:], (np.array(self.losses)[1:]-np.array(self.losses)[:-1])/np.array(self.losses)[1:])
        return fig, ax

