import matplotlib.pyplot as plt
from draw_neural_net import draw_neural_net

layer_sizes = [3, 5, 5, 4]
node_text = []
for i, num_nodes in enumerate(layer_sizes):
  layer_index = i + 1
  for j in range(num_nodes):
    node_index = j + 1
    node_text.append(r'$a^{(' + str(layer_index) + ')}_' + str(node_index) + '$')

fig = plt.figure(figsize=(5, 5))
ax = fig.gca()
draw_neural_net(ax, .1, .9, .1, .9, layer_sizes, node_text)

fig.savefig('../img/nn.png')