import os
from scipy.io.wavfile import read
import scipy.io.wavfile as wav
import subprocess as sp
import numpy as np
import argparse
import random
import os
import sys
from random import shuffle
import speechpy
import datetime

class AudioDataset():
    def __init__(self, file_path,transform=None):
        self.audio_dir = file_path
        self.transform = transform
        sound_file_path = "G:/Audio/ABSOLUTELY_00001.mp4.wav"
        try:
            with open(sound_file_path, 'rb') as f:
                riff_size, _ = wav._read_riff_chunk(f)
                file_size = os.path.getsize(sound_file_path)

        except OSError as err:
            print("OS error: {0}".format(err))
        except ValueError:
            print('file %s is corrupted!' % sound_file_path)


    def __getitem__(self,idx):
        # Get the sound file path
        sound_file_path = "G:/Audio/ABSOLUTELY_00001.mp4.wav"
        import soundfile as sf
        signal, fs = sf.read(sound_file_path)
        num_coefficient = 40
        frames = speechpy.processing.stack_frames(signal, sampling_frequency=fs, frame_length=0.02,frame_stride=0.02,zero_padding=True)
        power_spectrum = speechpy.processing.power_spectrum(frames, fft_points=2 * num_coefficient)[:, 1:]
        logenergy = speechpy.feature.lmfe(signal, sampling_frequency=fs, frame_length=0.02, frame_stride=0.02,num_filters=num_coefficient, fft_length=1024, low_frequency=0,high_frequency=None)
        sample = {'feature': logenergy, 'label': 1}
        if self.transform:
            sample = self.transform(sample)
        else:
            feature, label = sample['feature'], sample['label']
            sample = feature, label
        return sample


class CMVN(object):
    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']
        feature = speechpy.processing.cmvn(feature, variance_normalization=True)
        return {'feature': feature, 'label': label}


class Extract_Derivative(object):
    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']
        feature = speechpy.feature.extract_derivative_feature(feature)
        return {'feature': feature, 'label': label}


class Feature_Cube(object):

    def __init__(self, cube_shape):

        self.cube_shape = cube_shape
        if self.cube_shape != None:
            self.num_frames = cube_shape[0]
            self.num_features = cube_shape[1]
            self.num_channels = cube_shape[2]

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        if self.cube_shape != None:
            feature_cube = np.zeros((self.num_frames, self.num_features, self.num_channels), dtype=np.float32)
            feature_cube = feature[0:self.num_frames, :, :]
        else:
            feature_cube = feature

        # return {'feature': feature_cube, 'label': label}
        return {'feature': feature_cube[None, :, :, :], 'label': label}


class ToOutput(object):

    def __call__(self, sample):
        feature, label = sample['feature'], sample['label']

        return feature, label


class Compose(object):

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string


if __name__ == '__main__':
    dataset = AudioDataset(file_path="G:/Audio/ABSOLUTELY_00001.mp4.wav",transform=Compose([Extract_Derivative(), Feature_Cube(cube_shape=None), ToOutput()]))
    dataset
    idx = 0
    feature, label = dataset.__getitem__(idx)
    print(feature.shape)
    print(label)