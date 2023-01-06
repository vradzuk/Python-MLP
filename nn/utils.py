def train(prop, inputs, backprop, outputs):
    i, min_cost = 0, None
    while True:
        try:
            selected_input = inputs[i%len(inputs)]
            selected_output = outputs[i%len(outputs)]
            prop(selected_input)
            cost = backprop(selected_output)
            if min_cost is None or cost < min_cost:
                min_cost = cost
            print(f'\rCost: {min_cost:.8f}', end='')
            i += 1
        except KeyboardInterrupt:
            break
    print()
    
def test(prop, num_inputs=1):
    while True:
        try:
            inputs = [int(n) for n in input('Enter an input: ' if num_inputs==1 else
                                            f'Enter {num_inputs} inputs: ').split(' ')
                      if n.count('.') in (0, 1) and n.replace('.', '').isdigit()]
            if len(inputs) != num_inputs:
                print('Invalid input(s)')
                continue
            print('Result:', ' '.join(f'{r:.4f}' for r in prop(inputs)))
        except KeyboardInterrupt:
            return