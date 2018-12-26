import mnist_loader
import network
t_d, v_d, test_d = mnist_loader.load_data_wrapper()
print('training data')
print(type(t_d))
print(len(t_d))
print(t_d[0][0].shape)
print(t_d[0][1].shape)
#print(t_d[0])
print(len(test_d))
print(len(v_d))

net = network.Network([784, 40, 10])
net.SGD(t_d, 10, 10, 3.0, test_data=test_d)