
import heapq, itertools
import numpy as np




""" DEVS SCHEDULING """

class Event(object) :
    def __init__(self, callback ) :
        self._callback = callback
        self._active = True
        
    def callback(self) :
        return self._callback
    
    def isactive(self) :
        return self._active
    
    def activate(self) :
        raise Exception('task events cannot be re-activated')
    
    def deactivate(self) :
        self._active = False







class EventQ(object) :
    def __init__(self) :
        self._q = []
        
    def next(self) :
        self.top()      # burn any de-scheduled events
        time, event = heapq.heappop( self._q )
        return time, event
    
    def top(self) :
        while len( self._q ) > 0 :
            time, event = self._q[0]
            if event.isactive() :
                return ( time, event )
            else :
                heapq.heappop( self._q )
        return None
    
    def timeleft(self) :
        time, event = self.top()
        return time
    
    def advance(self, by_time=None ) :
        timeleft = self.timeleft()
        if by_time is None : by_time = timeleft
        #print by_time, timeleft
        
        assert by_time <= self.timeleft()
        
        for i, ( t, event ) in enumerate( self._q ) :
            self._q[i] = ( t - by_time, event )
            
        return by_time
    
    def schedule(self, callback, delay=0. ) :
        #if task in self._tasks : self.deschedule( task )
        event = Event( callback )
        heapq.heappush( self._q, ( delay, event ) )
        return event


""" very simple, just an EventQ and a time """
class Simulation(object) :
    def __init__(self) :
        self.eventq = EventQ()
        self.time = 0.
        
    def schedule(self, callback, delay=0. ) :
        return self.eventq.schedule( callback, delay )
        
    def get_time(self) :
        return self.time
    
    def get_next_action(self) :
        time, event = self.eventq.top()
        
        self.eventq.advance()
        self.time += time
        
        self.eventq.next()
        
        callback = event.callback()
        return callback

""" simulation methods """




