from graph_loader import GraphLoader


class BaseNavigator:
    graph = None

    def __init__(self, dataset_dir):
        if BaseNavigator.graph is None:
            BaseNavigator.graph = GraphLoader(dataset_dir).construct_graph()

        self.graph_state = None
        self.prev_graph_state = None
        self.initial_pano_id = None

    def navigate(self):
        raise NotImplementedError

    def get_heading_change(self):
        if self.prev_graph_state is None:
            return 0
        _, prev_heading = self.prev_graph_state
        _, cur_heading = self.graph_state

        a = cur_heading - prev_heading

        if -180 < a <= 180:
            a = a
        elif a > 180:
            a = a - 360
        elif a <= 180:
            a = a + 360

        return a / 180 # scale to -1 and 1

    def step(self, go_towards):
        '''
        Execute one step and update the state. 
        go_towards: ['forward', 'left', 'right']
        '''
        if not self.prev_graph_state:
            self.prev_graph_state = self.graph_state
        if go_towards == "stop":
            self.prev_graph_state = self.graph_state
            return
        next_panoid, next_heading = self._get_next_graph_state(self.graph_state, go_towards)

        if len(self.graph.nodes[next_panoid].neighbors) < 2:
            # stay still when running into the boundary of the graph
            #print(f'At the border (number of neighbors < 2). Did not go "{go_towards}".')
            return
        self.prev_graph_state = self.graph_state
        self.graph_state = (next_panoid, next_heading)
        
    def _get_next_graph_state(self, curr_state, go_towards):
        '''Get next state without changing the current state.'''
        curr_panoid, curr_heading = curr_state

        if go_towards == 'forward':
            neighbors = self.graph.nodes[curr_panoid].neighbors
            if curr_heading in neighbors:
                # use current heading to point to the next node
                next_node = neighbors[curr_heading]
            else:
                # weird node, stay put
                next_node = self.graph.nodes[curr_panoid]
        elif go_towards == 'left' or go_towards == 'right':
            # if turn left or right, stay at the same node 
            next_node = self.graph.nodes[curr_panoid]
        else:
            raise ValueError('Invalid action.')

        next_panoid = next_node.panoid
        next_heading = self._get_nearest_heading(curr_state, next_node, go_towards)
        return next_panoid, next_heading

    def _get_nearest_heading(self, curr_state, next_node, go_towards):
        _, curr_heading = curr_state
        next_heading = None

        diff = float('inf')
        if go_towards == 'forward':
            diff_func = lambda next_heading, curr_heading: 180 - abs(abs(next_heading - curr_heading) - 180)
        elif go_towards == 'left':
            diff_func = lambda next_heading, curr_heading: (curr_heading - next_heading) % 360
        elif go_towards == 'right':
            diff_func = lambda next_heading, curr_heading: (next_heading - curr_heading) % 360
        else:
            return curr_heading

        for heading in next_node.neighbors.keys():
            if heading == curr_heading and go_towards != 'forward':
                # don't match to the current heading when turning
                continue
            diff_ = diff_func(int(heading), int(curr_heading))
            if diff_ < diff:
                diff = diff_
                next_heading = heading

        if next_heading is None:
            next_heading = curr_heading
        return next_heading
