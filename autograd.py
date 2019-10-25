import numpy as np

class Tensor:
    def __init__(self, narray, requires_grad=False):
        self.narray = np.array(narray)
        self.op = ''
        self.requires_grad = requires_grad
        self.type = 'tensor'
        self.nodes = []
        self.grad = None

    def add(self, x):
        n = Tensor(self.narray + x.narray)
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'add'
        return n
    
    def sub(self, x):
        n = Tensor(self.narray - x.narray)
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'sub'
        return n

    def pow(self, x):
        n = Tensor(self.narray ** x.narray)
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'pow'
        return n

    def div(self, x):
        n = Tensor(self.narray // x.narray)
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'div'
        return n

    def matmul(self, x):
        n = Tensor(self.narray.dot(x.narray))
        n.nodes.append(self)
        n.nodes.append(x)
        n.type = 'op'
        n.op = 'matmul'
        return n

    def _op_backward(self):
        if self.type == "tensor":
            return Tensor([0])
        if self.op == 'sub':
            return Tensor([1])
        elif self.op == 'add':
            return Tensor([1])
        elif self.op == 'matmul':
            l = self.nodes[0]
            r = self.nodes[1]
            if l.requires_grad == True:
                return r
            return l
        elif self.op == 'pow':
            l = self.nodes[0]
            dl = Tensor([1.])
            r = self.nodes[1]
            return Tensor(r.narray * np.power(dl.narray, r.narray - 1))
        elif self.op == 'div':
            l = self.nodes[0]
            dl = Tensor([1.])
            r = self.nodes[1]
            dr = r._op_backward()
            return Tensor((dl.narray * r.narray + l.narray * dr.narray) / (r.narray ** 2))

    def backward(self, grad=None):
        prev = grad
        if grad is None:
            prev = self.narray
        grad = prev * self._op_backward().narray
        for i in range(len(self.nodes)):
            if self.nodes[i].requires_grad:
                self.nodes[i].grad = grad
            if self.nodes[i].type == "op":
                self.nodes[i].backward(grad)

x = Tensor([[1., 2., 3.]])
w = Tensor([[2.], [3.], [4.]], requires_grad=True)
b = Tensor([.0], requires_grad=True)
y_ = x.matmul(w).add(b)
y = Tensor([60.])
loss = y_.sub(y).pow(Tensor(2.)).div(Tensor(2.))

print(loss.narray)

loss.backward()

print(w.grad)
print(b.grad)

w.narray = w.narray + (0.001 * w.grad.T)
b.narray = b.narray + (0.001 * b.grad.T)

# print(w.narray)
# print(b.narray)

# xs = []
# ys = []
# for i in range(10):
#     xs.append(Tensor([[i]]))
#     ys.append(Tensor([[i * 2]]))

# print(xs)
# print(ys)
# w = Tensor([[2.]], requires_grad=True)
# b = Tensor([.0], requires_grad=True)

# for i in range(10):
#     for j in range(len(xs)):
#         x = xs[j]
#         y = ys[j]
#         y_ = x.matmul(w).add(b)
#         print('x:')
#         print(x.narray)
#         print('y_:')
#         print(y_.narray)
#         loss = y_.sub(y).pow(Tensor(2)).div(Tensor(2))
#         print('loss:')
#         print(loss.narray)
#         loss.backward()
    
#         w.narray = w.narray + (0.001 * w.grad.T)
#         b.narray = b.narray + (0.001 * b.grad.T)

# print(w.narray)