# I downloaded this recipe from PyGame wiki... from ASPN Cookbook 2005, apparently.

# to make a "channel", just make a signal that isn't "owned" by a particular entity

from weakref import *
import inspect

class Message(object) :
    def __init__(self, call, *args, **kwargs ) :
        self.call = call
        self.args = args
        self.kwargs = kwargs
        
    def send(self) :
        self.call( *self.args, **self.kwargs )
        
    def __call__(self) : self.send()


class Signal(object) :
    def __init__(self):
        self.slots = []
 
        # for keeping references to _WeakMethod_FuncHost objects.
        # If we didn't, then the weak references would die for
        # non-method slots that we've created.
        self.funchost = []
 
    def __call__(self, *args, **kwargs):
        for i, slot in enumerate(self.slots):
            if slot != None:
                slot(*args, **kwargs)
            else:
                del self.slots[i]
                
    def call(self, *args, **kwargs):
        self.__call__(*args, **kwargs)
 
    def connect(self, slot):
        self.disconnect(slot)
        if inspect.ismethod(slot):
            self.slots.append(WeakMethod(slot))
        else:
            o = _WeakMethod_FuncHost(slot)
            self.slots.append(WeakMethod(o.func))
            # we stick a copy in here just to keep the instance alive
            self.funchost.append(o)
 
    def disconnect(self, slot):
        try:
            for i, wm in enumerate(self.slots):
                if inspect.ismethod(slot):
                    if wm.f == slot.im_func and wm.c() == slot.im_self:
                        del self.slots[i]
                        return
                else:
                    if wm.c().hostedFunction == slot:
                        del self.slots[i]
                        return
        except:
            pass
 
    def disconnectAll(self):
        self.slots = []
        self.funchost = []
 
class _WeakMethod_FuncHost:
    def __init__(self, func):
        self.hostedFunction = func
    def func(self, *args, **kwargs):
        self.hostedFunction(*args, **kwargs)
 
# this class was generously donated by a poster on ASPN (aspn.activestate.com)
class WeakMethod:
    def __init__(self, f):
        self.f = f.im_func
        self.c = ref(f.im_self)
    def __call__(self, *args, **kwargs):
        if self.c() == None : return
        self.f(self.c(), *args, **kwargs)
        
        