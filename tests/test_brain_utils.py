from brain import Brain
from utils import value_or_default


class TestBrainUtils(object):

    def __init__(self, default_p=0.01, default_area_size=1000, default_winners_size=300,
                 default_stimulus_size=300, beta=0.01):
        self.P = default_p
        self.area_size = default_area_size
        self.winners_size = default_winners_size
        self.stimulus_size = default_stimulus_size
        self.beta = beta
        self._init_data()

    def __getattr__(self, item):
        if item.startswith('area'):
            area_index = int(item[4:])
            return self.brain.areas[self._areas[area_index]]
        if item.startswith('stim'):
            stimulus_index = int(item[4:])
            return self.brain.stimuli[self._stimuli[stimulus_index]]

    def _init_data(self):
        self.brain = None
        # Area names, ordered by their creation time
        self._areas = []
        # Stimuli names, order by their creation time
        self._stimuli = []

    def create_brain(self, number_of_areas, p=None, area_size=None, winners_size=None, beta=None):
        self._init_data()

        self.brain = Brain(value_or_default(p, self.P))
        for i in range(1, number_of_areas + 1):
            self._add_area(str(i), area_size, winners_size, beta)
        return self.brain

    def create_and_stimulate_brain(self, number_of_areas, number_of_stimulated_areas=1,
                                   stimulus_size=None, p=None, area_size=None, winners_size=None, beta=None):
        assert number_of_stimulated_areas <= number_of_areas

        self.create_brain(number_of_areas=number_of_areas, p=p, area_size=area_size,
                          winners_size=winners_size, beta=beta)
        self._add_stimulus('1', stimulus_size)

        areas_to_stimulate = self._areas[:number_of_stimulated_areas]
        self.brain.project(area_to_area={}, stim_to_area={'1': areas_to_stimulate})
        return self.brain

    def _add_area(self, name, n, k, beta):
        self.brain.add_area(name=name,
                            n=value_or_default(n, self.area_size),
                            k=value_or_default(k, self.winners_size),
                            beta=value_or_default(beta, self.beta))
        self._areas.append(name)

    def _add_stimulus(self, name, k):
        self.brain.add_stimulus(name=name,
                                k=value_or_default(k, self.stimulus_size))
        self._stimuli.append(name)
