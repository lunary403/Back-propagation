import tkinter as tk
from tkinter import *
from BP_imp import *



X_train_list = X_train.values.tolist()
X_test_list = X_test.values.tolist()
y_train_list = y_train.values.tolist()
y_test_list = y_test.values.tolist()

def submit():
    act_func = activation_radioButton.get()
    if act_func == 1:
        hidden_layers = hidden_layers_num.get()
        neurons_in_layer = neurons_per_layer.get()
        learning_rate = LR.get()
        epochs = epochs_num.get()

        bias = bias_Checkbutton.get()

        network = initialize_network(5, neurons_in_layer,3, hidden_layers, bias)
        train_network_sigmoid(network, X_train_list, learning_rate, epochs, 3)


    elif act_func == 2:
        hidden_layers = hidden_layers_num.get()
        neurons_in_layer = neurons_per_layer.get()
        learning_rate = LR.get()
        epochs = epochs_num.get()

        bias = bias_Checkbutton.get()

        network = initialize_network(5, neurons_in_layer,3, hidden_layers, bias)
        train_network_tanh(network, X_train_list, learning_rate, epochs, 3)








window = Tk()

window.geometry('700x500+250+10')
window.title('Dry Beans')




tk.Label(text='Enter number of hidden Layers :').place(x=0, y=40)
tk.Label(text='Enter number of neurons in each hidden layers : ').place(x=0, y=110)
tk.Label(text='Enter learning rate (eta) : ').place(x=0, y=180)
tk.Label(text='Enter number of epochs : ').place(x=0, y=250)


hidden_layers_num = IntVar()
tk.Entry(textvariable=hidden_layers_num).place(x=10, y=70)
neurons_per_layer = IntVar()
tk.Entry(textvariable=neurons_per_layer).place(x=10, y=140)
LR = DoubleVar()
tk.Entry(textvariable=LR).place(x=10, y=210)
epochs_num = IntVar()
tk.Entry(textvariable=epochs_num).place(x=10, y=280)


tk.Label(text='Select an activation function :').place(x=0, y=320)

activation_radioButton = tk.IntVar()
# tanh_radioButton = tk.IntVar()

tk.Radiobutton(window, text="Sigmoid", variable=activation_radioButton, value=1).place(x=10, y=350)
tk.Radiobutton(window, text="tanh", variable=activation_radioButton, value=2).place(x=80, y=350)


tk.Label(text='Select whether you want bias in data or not :').place(x=0, y=390)
bias_Checkbutton = tk.IntVar()
tk.Checkbutton(text="bias", variable=bias_Checkbutton, onvalue=1, offvalue=0).place(x=120, y=410)

tk.Button(text='Submit', width=22, height=1, cursor='hand2', bd=3, command=submit).place(x=250, y=460)



