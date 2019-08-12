import os
import yaml

class Config(dict):
    def __init__(self, opts, mode):
        with open(opts.config, 'r') as f:
            self._yaml = f.read()
            self._dict = yaml.load(self._yaml)

        self.modify_param(opts, mode)

 
    def __getattr__(self, name):
        if self._dict.get(name) is not None:
            return self._dict[name]
        return None

    def print(self):
        print('Model configurations:')
        print('---------------------------------')
        print(self._yaml)
        print('')
        print('---------------------------------')
        print('')


    def modify_param(self, opts, mode):
        self._dict['PATH'] = opts.path
        self._dict['NAME'] = opts.name
        self._dict['RESUME_ALL'] = opts.resume_all

        if mode == 'test':
            self._dict['DATA_TEST_GT'] = opts.input
            self._dict['DATA_TEST_MASK'] = opts.mask
            self._dict['DATA_TEST_STRUCTURE'] = opts.structure
            self._dict['DATA_TEST_RESULTS'] = opts.output
            self._dict['MODEL'] = opts.model
