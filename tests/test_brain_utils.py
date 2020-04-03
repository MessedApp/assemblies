from brain import Brain
from utils import value_or_default


class TestBrainUtils(object):

    def __init__(self, default_p=0.1, default_area_size=10000, default_winners_size=100,
                 default_stimulus_size=100, beta=0.1):
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
        self.output_area = None
        # Stimuli names, order by their creation time
        self._stimuli = []

    def create_brain(self, number_of_areas, p=None, area_size=None, winners_size=None, beta=None,
                     add_output_area=False):
        self._init_data()

        self.brain = Brain(value_or_default(p, self.P))
        for i in range(1, number_of_areas + 1):
            self._add_area(str(i), area_size, winners_size, beta)

        if add_output_area:
            self.brain.add_output_area('output')
            self.output_area = self.brain.output_areas['output']
        return self.brain

    def create_and_stimulate_brain(self, number_of_areas, number_of_stimulated_areas=1,
                                   stimulus_size=None, p=None, area_size=None, winners_size=None, beta=None,
                                   add_output_area=False):
        assert number_of_stimulated_areas <= number_of_areas

        self.create_brain(number_of_areas=number_of_areas, p=p, area_size=area_size,
                          winners_size=winners_size, beta=beta, add_output_area=add_output_area)
        self._add_stimulus('stimulus', stimulus_size)

        areas_to_stimulate = self._areas[:number_of_stimulated_areas]
        self.brain.project(area_to_area={}, stim_to_area={'stimulus': areas_to_stimulate})
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
