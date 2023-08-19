"""A Discrete Event Simulation microframework with fairly faithful coupling in these sense of [1].

Additionally, the dynamics of "Procedure"-type automata can be defined in a more straightforward way
using coroutines.

[1] https://en.wikipedia.org/wiki/DEVS

"""
from dataclasses import dataclass, field
from typing import List, Any, Optional, Set
from typing import Callable, TypeVar, Tuple, Dict, Union
import heapq  # TODO: Use priority queue to speed up dispatch.
from uuid import uuid4
import pandas as pd
import numpy as np
import scipy as sp
import scipy.stats
from abc import ABCMeta, abstractmethod
from functools import wraps

import logging

LOG = logging.getLogger(__name__)


def place_of(x):
    yield x


def singleton(factory):
    return factory()


closure = singleton  # alias


class Automata(metaclass=ABCMeta):

    @abstractmethod
    def timeout(self) -> Optional[float]:
        """How long until automata state expires?

        None means it does not expire.

        """

    @abstractmethod
    def expire(self) -> Any:
        """Execute the automata endogeneous transition. And we can obtain output.

        We may assume the automata has been advanced to its expiration time.

        """

    @abstractmethod
    def advance(self, duration: float):
        """Inform the automata that time has passed."""

    @abstractmethod
    def put(self, msg: Any):
        """Insert a symbol/msg into the automata."""


M1 = TypeVar("M1")
M2 = TypeVar("M2")
RoutingFn = Callable[[M1], Tuple[Automata, M2]]


class Simulation:

    def __init__(self):
        self.components: List[Automata] = []
        self.routes: List[Tuple[Automata, RoutingFn]] = []

    def timeout(self) -> Optional[float]:
        result = None
        for a in self.components:
            dt = a.timeout()
            if dt is None:
                continue
            if result is None or dt < result:
                result = dt
        return result

    def flush(self):
        while True:
            ready = self.ready()
            if len(ready) < 1:
                return

            for c in ready:
                msg = c.expire()
                route_fn, = (fn for (c_, fn) in self.routes if c_ is c)  # find and ensure singleton
                targ, msg_ = route_fn(msg)
                targ.put(msg_)

    def ready(self):
        return [
            c
            for c in self.components
            for dt in place_of(c.timeout())
            if dt is not None and dt <= 0.
        ]

    def simulate(self, t: float):
        if t <= 0.:
            return

        while True:
            self.flush()
            dt = self.timeout()
            if dt is None or t < dt:
                self.advance(t)
                return
            # else
            self.advance(dt)
            t -= dt

    def advance(self, t: float):
        """

        t _must_ be no greater than dt!

        """
        LOG.debug(f"advancing by {t}")
        for c in self.components:
            c.advance(t)

    def add_component(self, component: Automata):
        return self.components.append(component)

    def add_route(self, producer: Automata, routing_fn: RoutingFn):
        self.routes.append((producer, routing_fn))


class ProcedureAutomata(Automata, metaclass=ABCMeta):
    _timeout: float

    def timeout(self) -> Optional[float]:
        return self._timeout

    def advance(self, duration: float):
        if self._timeout is not None:
            self._timeout -= duration

    def expire(self) -> Any:
        msg, self.msg = self.msg, None
        LOG.debug(f"{self} expiring;")
        LOG.debug(f"{self} -> {msg}")
        self._next()
        return msg

    def _next(self):
        try:
            directive = next(self._c)
        except StopIteration:
            self._timeout = None
        self._consume(directive)

    def _consume(self, directive):
        LOG.debug(f"{self}, next: {directive}")
        if isinstance(directive, self.emit):
            self.msg = directive.msg
            self._timeout = directive.delay

        elif isinstance(directive, self.await_input):
            self._timeout = None

        else:
            raise Exception("unrecognized")

    def put(self, msg: Any):
        LOG.debug(f"{msg} -> {self}")
        try:
            directive = self._c.send(msg)
        except StopIteration:
            self._timeout = None
        self._consume(directive)

    def __init__(self):
        self._c = self.co()
        self._next()

    @abstractmethod
    def co(self):
        """Enjoy..."""

    @dataclass(frozen=True)
    class emit:
        msg: Any
        delay: float = field(default=0.)

    @dataclass(frozen=True)
    class await_input:
        pass


# END FRAMEWORK CODE; BEGIN LIBRARY CODE


@dataclass
class Clock(Automata):
    # params
    period: float
    initial_timeout: float = field(default=0.)

    # state
    _timeout: float = field(init=False)

    def __post_init__(self):
        self._timeout = self.initial_timeout

    def timeout(self) -> Optional[float]:
        return self._timeout

    def advance(self, duration: float):
        self._timeout -= duration

    def put(self, msg: Any):
        raise Exception("pure emitter")

    @dataclass(frozen=True)
    class Tick:
        pass

    def expire(self) -> Tick:
        self._timeout = self.period
        return self.Tick()


class Accum(Automata):

    def __init__(self):
        self.count = 0

    def timeout(self) -> Optional[float]:
        return None

    def expire(self) -> Any:
        raise Exception("does not expire.")

    def advance(self, duration: float):
        pass

    def put(self, n: Any):
        self.count += n


class Timer(Automata):

    def __init__(self):
        self.elapsed = 0

    def timeout(self) -> Optional[float]:
        return None

    def expire(self) -> Any:
        raise Exception("does not expire.")

    def advance(self, duration: float):
        self.elapsed += duration

    def put(self, n: Any):
        raise Exception("does not take input.")


class Sink(ProcedureAutomata):

    def __init__(self):
        super().__init__()
        self.collection = []

    def co(self):
        while True:
            item = yield self.await_input()
            self.collection.append(item)


if False:
    # little demo.
    sim = Simulation()

    clock = Clock()
    accum = Accum()
    timer = Timer()

    sim.add_component(clock)
    sim.add_component(accum)
    sim.add_component(timer)

    def route_clock(n):
        return accum, n

    sim.add_route(clock, route_clock)

    sim.simulate(13.37)


def max_rates(c: np.ndarray, Xbar: float, X: np.ndarray) -> np.ndarray:
    """This is the "progressive filling algorithm".

    https://en.wikipedia.org/wiki/Max-min_fairness#Progressive_filling_algorithm

    Args:
        c: target rates
        Xbar: output quota; could be np.inf
        X: availability of inputs

    """
    nc, = c.shape
    nX, = X.shape
    assert nX == nc
    assert np.all(c >= 0.) and np.all(X >= 0.)

    x = np.zeros_like(c)
    while True:
        I = (c > 0.) & (x < X)  # we project to the subspace where progress can still be made
        alpha1 = np.min((X[I] - x[I]) / c[I])  # distances to remaining availabilities
        alpha2 = (Xbar - np.sum(x)) / np.sum(c[I])  # distance to quota
        alpha = min(alpha1, alpha2)
        if alpha <= 0.:  # no more progress is possible
            return x
        x[I] += alpha * c[I]


def df_multinomial(distr: sp.stats.rv_discrete, bucket_df: pd.DataFrame, frac_col: str, n_col: str):
    n = distr.ppf(np.random.random())  # sample distr
    parts = bucket_df[frac_col]
    parts /= parts.sum()  # normalize.
    ns = pd.Series(
        np.random.multinomial(n, parts.to_numpy()),
        parts.index
    )
    return bucket_df.assign(**{n_col: ns}).drop(frac_col, axis=1)


def create_batch(num_df: pd.DataFrame, n_col: str, id_col: str):

    def gen():
        for _, row in num_df.iterrows():
            n = row[n_col]
            row_ = row.drop(n_col)
            for _ in range(n):
                record = row_.to_dict()
                record[id_col] = uuid4()
                yield record

    return pd.DataFrame.from_records(gen())


if __name__ == "__main__":

    sim = Simulation()

    DIMS = "locale exp tg slot domain source".split()
    ID_COL = "id"
    PRIORITY = "prio"

    SEC_PER_DAY = 24. * 60. * 60.  # sec/day

    # create a source
    @dataclass
    class DataProcess(Automata):

        # params
        sample_batch_fn: Callable[[float, float], pd.DataFrame]
        start_time: float = 0.

        # state
        elapsed: float = field(init=False, default=0.)
        current: Union[None, pd.DataFrame] = field(init=False, default=None)
        output: Union[None, pd.DataFrame] = field(init=False, default=None)

        def __post_init__(self):
            self.current = self.sample_batch_fn(0., 0.)

        def advance(self, duration: float):
            t1 = self.elapsed  # previous time
            t2 = self.elapsed = t1 + duration
            df = self.sample_batch_fn(t1, t2)
            df = df.assign(
                timestamp=sorted(np.random.uniform(t1, t2, len(df)))
            )
            if self.current is None:
                self.current = df
            else:
                self.current = self.current.append(df, ignore_index=True)

        @dataclass(frozen=True)
        class Emit:
            """Release a batch."""

        def put(self, _trigger: Emit):
            if self.output is not None:
                raise Exception("you almost lost output...")
            self.output, self.current = self.current, None

        def timeout(self):
            return 0. if self.output is not None else None

        def expire(self) -> pd.DataFrame:
            if self.output is None:
                raise Exception("no output")

            result, self.output = self.output, None
            return result


    @closure
    class dp_params:

        rate_per_day = 100

        # proportions of the population in different combinations of attributes
        buckets_df = pd.DataFrame.from_records(
            dict(nationality=nationality, religion=religion, other="some_constant", portion=1)
            for nationality in ["USA", "there are other countries?"]
            for religion in ["christian", "jewish", "muslim", "etc"]
        )

        def dp_sample(self, t1, t2):
            duration = t2 - t1
            rate_per_sec = self.rate_per_day / SEC_PER_DAY
            rate = rate_per_sec * duration

            nums_df = df_multinomial(sp.stats.poisson(rate), self.buckets_df, "portion", "n")
            return create_batch(nums_df, "n", "id")

    dp = DataProcess(dp_params.dp_sample)
    sim.add_component(dp)

    class Workflow(ProcedureAutomata):
        # params
        period: float
        initial_timeout: float

        # state
        elapsed: float
        last_run: float

        def __init__(self, period: float, initial_timeout: float = None):
            self.period = period
            self.initial_timeout = initial_timeout or period

            # state
            self.elapsed = 0.
            self.last_run = self.initial_timeout - period

            super().__init__()

        def advance(self, duration: float):
            self.elapsed += duration
            super().advance(duration)

        @dataclass(frozen=True)
        class Want:  # a batch
            pass

        def co(self):
            yield self.emit(self.Want(), self.initial_timeout)
            while True:
                batch: pd.DataFrame = yield self.await_input()

                # Add some relevant temporal info.
                updates = batch.assign(
                    batch_timestamp=self.elapsed,
                    fake_timestamp=np.random.uniform(self.last_run, self.elapsed, len(batch)),
                )
                self.last_run = self.elapsed  # oh.. interesting...

                yield self.emit(updates)
                yield self.emit(self.Want(), self.period)

    w = Workflow(SEC_PER_DAY)
    sim.add_component(w)

    def route_data(msg):
        return w, msg

    sim.add_route(dp, route_data)

    @dataclass
    class Table(Automata):
        data: pd.DataFrame = None

        @dataclass(frozen=True)
        class Insert:
            data: pd.DataFrame

        def put(self, msg: Any):

            if isinstance(msg, self.Insert):
                self.data = (
                    msg.data if self.data is None
                    else self.data.append(msg.data, ignore_index=True)
                )

        def timeout(self) -> Optional[float]:
            return None

        def expire(self) -> Any:
            raise RuntimeError("Table has no self-transitions.")

        def advance(self, duration: float):
            """Do nothing."""

    table = Table()
    sim.add_component(table)

    def route_workflow(msg):
        LOG.debug(f"routing {msg}")
        if isinstance(msg, Workflow.Want):
            return dp, dp.Emit()

        if isinstance(msg, pd.DataFrame):
            return table, table.Insert(msg)

    sim.add_route(w, route_workflow)

    # Everybody be like "use Python logging".
    # But there's a lot of room (specifically with logging.basicConfig)
    # for having library code eff up global state in a way that's annoying to recover from
    # logging.basicConfig(level=logging.ERROR, force=True)  # force not working?
    LOG.setLevel(logging.WARNING)
    sim.simulate(5.5 * SEC_PER_DAY)

    if True:
        import matplotlib.pyplot as plt

        df = table.data
        plt.scatter(df.timestamp, df.fake_timestamp)
        plt.show()
