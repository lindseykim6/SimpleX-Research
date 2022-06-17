from _dataloader2 import *
from _networks import *
from _iterative_proj import *

MODEL_DIR = "models/"
NAME = "sphere"
NUM_PARTICLES = 1
DIMENSION = 2
NUM_ITER = 5
C_LAYERS = [64, 32, 1]

def load_model():
    model_path = MODEL_DIR + NAME + "/" + "best_model.pt"

    func_net = Simple_NN(num_particles=NUM_PARTICLES,
                           dimension=DIMENSION, num_features=C_LAYERS)
    func_net.load_state_dict(torch.load(model_path))
    proj_model = Projection(num_particles=NUM_PARTICLES,
                            dimension=DIMENSION, func=func_net, num_iter=NUM_ITER).cpu()
    return proj_model

def test_projection():
    model = load_model()
    d, l = generate_2dim(200)
    pred = model(d)
    pred2 = model(pred)
    pred = pred.detach().numpy()
    d = d.detach().numpy()
    l = l.detach().numpy()
    plt.scatter(d[:,0,0], d[:,0,1], c = 'y')
    plt.scatter(l[:,0,0], l[:,0,1], c = 'g')
    plt.scatter(pred[:,0,0], pred[:,0,1], c = 'b')
    plt.show()
    
def show_implicit_function():
    model = load_model()
    cons = model.func
    x_ = np.arange(-1.5, 1.5, 0.01)
    y_ = np.arange(-1.5, 1.5, 0.01)
    x_, y_ = np.meshgrid(x_, y_)
    x = torch.Tensor(x_).view(x_.size, 1)
    y = torch.Tensor(y_).view(y_.size, 1)
    data = torch.cat((x, y), 1).view(x.size()[0], 1, 2)
    c = cons(data)
    c = c.view(x_.shape[0], x_.shape[1]).detach().numpy()

    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_, y_, c, cmap = 'rainbow')
    plt.show()    

if __name__ == '__main__':
    test_projection()
    show_implicit_function()