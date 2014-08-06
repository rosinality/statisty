"""
A simple progress bar to monitor MCMC sampling progress.
Modified from original code by Corey Goldberg (2010), and pymc code.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import
from future.builtins import super
from future.builtins import int
from future.builtins import round
from future import standard_library
standard_library.install_hooks()

import sys
import time
import uuid
from statisty.utils import is_ipython
try:
    from IPython.core.display import HTML, Javascript, display
except ImportError:
    pass

__all__ = ['progress_bar']


class ProgressBar(object):
    def __init__(self, iterations, animation_interval=.5):
        self.iterations = iterations
        self.start = time.time()
        self.last = 0
        self.animation_interval = animation_interval

    def percentage(self, i):
        return 100 * i / float(self.iterations)

    def update(self, i):
        elapsed = time.time() - self.start
        i += 1

        if elapsed - self.last > self.animation_interval:
            self.animate(i + 1, elapsed)
            self.last = elapsed
        elif i == self.iterations:
            self.animate(i, elapsed)


class TextProgressBar(ProgressBar):
    def __init__(self, iterations, printer):
        self.fill_char = '-'
        self.width = 40
        self.printer = printer

        super(TextProgressBar, self).__init__(iterations)
        self.update(0)

    def animate(self, i, elapsed):
        self.printer(self.progbar(i, elapsed))

    def progbar(self, i, elapsed):
        bar = self.bar(self.percentage(i))
        return "[%s] %i of %i complete in %.1f sec" % (bar, i, self.iterations, round(elapsed, 1))

    def bar(self, percent):
        all_full = self.width - 2
        num_hashes = int(percent / 100 * all_full)

        bar = self.fill_char * num_hashes + ' ' * (all_full - num_hashes)

        info = '%d%%' % percent
        loc = (len(bar) - len(info)) // 2
        return replace_at(bar, info, loc, loc + len(info))


def replace_at(str, new, start, stop):
    return str[:start] + new + str[stop:]


def consoleprint(s):
    if sys.platform.lower().startswith('win'):
        print(s, '\r', end='')
    else:
        print(s)


def ipythonprint(s):
    print('\r', s, end='')
    sys.stdout.flush()

def progress_bar(iters):
    if is_ipython():
        return TextProgressBar(iters, ipythonprint)
    else:
        return TextProgressBar(iters, consoleprint)
