import numpy as np
import torch


class LocalTransform():
    r'''
    Init args:
        tranform: linear of sinusoidal
        period: sequential length of inputs
    Call args:
        input: shape=[batch_size=b, inp_size=I], values in [0,1]
        iteration: within period
    '''
    def __init__(self, period, transform='sin'):
        super(LocalTransform, self).__init__()
        self.transform = transform
        self.period = period

    def __call__(self, iteration, input):
        iteration = iteration % self.period
        
        if self.transform == 'lin':
            factor = iteration/self.period
            return input*factor
        
        elif self.transform == 'sin':
            factor=np.sin(iteration*(np.pi/self.period))
            return input*factor

class SinTransform():
    r'''
    Init args:
        tranform: linear of sinusoidal
        period: sequential length of inputs
    Call args:
        input: shape=[batch_size=b, inp_size=I], values in [0,1]
        iteration: within period
    '''
    def __init__(self, freq, phase, amplitude):
        super(SinTransform, self).__init__()
        self.period = 784
        self.freq = freq
        self.phase = phase
        self.amplitude = amplitude

    def __call__(self, iteration, input):
        iteration = iteration % self.period
        factor = self.amplitude*np.sin(iteration*(self.freq*2*np.pi/self.period)+self.phase)
        factor += 1 # shift
        
        output = torch.where(input > 0, input*factor, input)
        return output

class StepTransform():
    r'''
    Init args:
        step_size: scalar
        step_length: sequential length inputs with *= (1+stepsize)
    Call args:
        input: shape=[batch_size=b, inp_size=I], values in [0,1]
        iteration: within period
    '''
    def __init__(self, step_size, step_length, step_position):
        super(StepTransform, self).__init__()
        self.period = 784
        self.step_size = step_size
        self.step_position = step_position % self.period
        self.step_length = min(step_length, self.period)

    def get_factor(self, iteration):
        iteration = iteration % self.period
        if iteration > self.step_position and iteration < self.step_position+self.step_length :
            factor = 1+self.step_size
        else:
            factor = 1
        return factor

    def __call__(self, iteration, input):
        factor = self.get_factor(iteration)
        output = torch.where(input > 0, input*factor, input)
        return output

