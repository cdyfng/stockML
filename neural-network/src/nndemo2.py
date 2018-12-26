import mnist_loader
import network2
t_d, v_d, test_d = mnist_loader.load_data_wrapper()
print('training data')
print(type(t_d))
print(len(t_d))
print(t_d[0][0].shape)
print(t_d[0][1].shape)
#print(t_d[0])
print(len(test_d))
print(len(v_d))

net = network2.Network([784, 40, 10], cost=network2.CrossEntropyCost)
net.SGD(t_d, 10, 10, 3.0, evaluation_data=test_d, monitor_evaluation_accuracy=True)