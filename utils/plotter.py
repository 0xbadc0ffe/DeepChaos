import matplotlib.pyplot as plt
import os


def _load(path=None):
    if path is None:
        data = save_handler.load_cfgs()
        plots = save_handler.load_plot_data()
    else:
        data = save_handler.load_cfgs(path=path)
        plots = save_handler.load_plot_data(path=path)
    if plots is None or data is None:
        raise Exception("Model not saved")
    return data, plots

def plot_loss(plots=None, show=True, plot_name=["loss", "weigths_norm"]): 
    if plots is None:
        _, plots = _load() 


    try:
        if type(plot_name) is list: 
            if  len(plot_name)>1:
                loss_plotter = plots[plot_name[0]]
                weigths_norm_plt = plots[plot_name[1]] 
                plt.plot(weigths_norm_plt, label="weigths norm")
            else:
                loss_plotter = plots[plot_name[0]]
        else:
            loss_plotter = plots[plot_name]
    except KeyError:
        return False
    # Loss
    plt.plot(loss_plotter, label="loss")
    plt.title(f"Loss value")
    plt.xlabel("t")
    plt.ylabel(f"L(t)")
    plt.legend()
    if show:
        plt.show()
    return True

def plot_err(plots=None, show=True, plot_name=["error_1-for", "error_n-for", "error_t-for"], threshold=0.2, washout=25): 
    if plots is None:
        _, plots = _load() 
   
    try:
        error_plot_1for = plots[plot_name[0]]
        error_plot_nfor = plots[plot_name[1]]
        error_plot_tfor = plots[plot_name[2]]
    except KeyError:
        return False

    plt.plot(error_plot_1for, label="Err-1-for", color="red")
    plt.plot(error_plot_nfor, label="Err-n-for", color="green")
    plt.plot(error_plot_nfor, label="Err-t-for", color="blue")
    plt.title(f"Forcasting Error and Predictability Horizon")
    plt.xlabel("t")
    plt.ylabel(f"E(t)")
    leng = len(error_plot_1for)-1
    for k,v in enumerate(error_plot_1for):
        if (k>=washout and v>threshold) or k==leng:
            plt.scatter(k, error_plot_1for[k], color="red") # Predictability Horizon for 1-for
            break
    leng = len(error_plot_nfor)-1
    for k,v in enumerate(error_plot_nfor):
        if (k>=washout and v>threshold) or k==leng:
            plt.scatter(k, error_plot_nfor[k], color="green") # Predictability Horizon for n-for
            break
    leng = len(error_plot_tfor)-1
    for k,v in enumerate(error_plot_tfor):
        if (k>=washout and v>threshold) or k==leng:
            plt.scatter(k, error_plot_tfor[k], color="blue") # Predictability Horizon for t-for
            break
    plt.legend()
    if show:
        plt.show()
    return True

def plot_component(k=0, plots=None, data = None, lines={"gt":True, "1-for":True, "n-for":True, "t-for":True}, show=True):
    if plots is None:
        data, plots = _load()
    elif data is None:
        data = save_handler.load_cfgs()

    try:
        for_hor = data["for_hor"]
        Nt = data["Nt"]     

        plt.title(f"X{k} Forecasting")
        if lines["gt"]:
            x_sys = plots["ground_truth_sys"]
            if x_sys.shape[1] == 1:
                plt.plot(x_sys[:], color="blue", label="ground truth")
            else:
                plt.plot(x_sys[:,k], color="blue", label="ground truth")   
        if lines["1-for"]:
            x_for_1 = plots["1-forecasting"]
            if x_for_1.shape[1] == 1:
                plt.plot(x_for_1[:], color="red", label="1-forecasting")
            else:
                plt.plot(x_for_1[:,k], color="red", label="1-forecasting")   
        if lines["n-for"]:
            x_for_n = plots["n-forecasting"]
            if x_for_n.shape[1] == 1:
                plt.plot(x_for_n[:], color="green", label="n-forecasting")
            else:
                plt.plot(x_for_n[:,k], color="green",label="n-forecasting")
        if lines["t-for"]:
            x_for_t = plots["trained-forecasting"]
            if x_for_t.shape[1] == 1:
                plt.plot(x_for_t[:], color="violet", label=f"{for_hor}-forecasting (trained)")
            else:
                plt.plot(x_for_t[:,k], color="violet",label=f"{for_hor}-forecasting (trained)")
    except KeyError:
        return False

    plt.xlabel("t")
    plt.ylabel(f"x{k}(t)")
    plt.legend()
    if x_sys.shape[1] == 1:
        plt.scatter(Nt, x_sys[Nt], color="black") # training horizon position
    else:
        plt.scatter(Nt, x_sys[Nt, k], color="black") # training horizon position
    if show:
        plt.show()
    return True

def plot3d(plots=None, data=None, lines={"gt":True, "1-for":False, "n-for":False, "t-for":True}, show=True):
    if plots is None:
        data, plots = _load()
    elif data is None:
        data = save_handler.load_cfgs()

    try:
        Nt = data["Nt"]
        ax = plt.axes(projection='3d')
        if lines["t-for"]:
            x_for_t = plots["trained-forecasting"]
            if x_for_t.shape[1] <3 : 
                return
            ax.plot3D(x_for_t[:,0].numpy(), x_for_t[:,1].numpy(), x_for_t[:,2].numpy(), 'violet', label="t-for")
        if lines["gt"]:
            x_sys = plots["ground_truth_sys"]
            ax.plot3D(x_sys[:,0].numpy(), x_sys[:,1].numpy(), x_sys[:,2].numpy(), 'blue', label="gnd")
        if lines["1-for"]:
            x_for_1 = plots["1-forecasting"]
            ax.plot3D(x_for_1[:,0].numpy(), x_for_1[:,1].numpy(), x_for_1[:,2].numpy(), 'red', label="1-for")
        if lines["n-for"]:
            x_for_n = plots["n-forecasting"]
            ax.plot3D(x_for_n[:,0].numpy(), x_for_n[:,1].numpy(), x_for_n[:,2].numpy(), 'green', label="n-for")
    except KeyError:
        return False

    plt.title("3D plot")
    ax.legend()
    ax.scatter(x_sys[0,0],x_sys[0,1],x_sys[0,2], color="green") # initial position
    ax.scatter(x_sys[Nt,0],x_sys[Nt,1],x_sys[Nt,2], color="black") # training horizon position
    if show:
        plt.show()
    return True

def plot2d(plots=None, data=None, lines={"gt":True, "1-for":False, "n-for":False, "t-for":True}, show=True):
    if plots is None:
        data, plots = _load()
    elif data is None:
        data = save_handler.load_cfgs()

    try:
        for_hor = data["for_hor"]
        Nt = data["Nt"]     

        plt.title(f"X0 vs X1")
        if lines["gt"]:
            x_sys = plots["ground_truth_sys"]
            if x_sys.shape[1] < 2:
                return
            plt.plot(x_sys[:,0], x_sys[:,1], color="blue", label="ground truth")   
        if lines["1-for"]:
            x_for_1 = plots["1-forecasting"]
            plt.plot(x_for_1[:,0], x_for_1[:,1], color="red", label="1-forecasting")   
        if lines["n-for"]:
            x_for_n = plots["n-forecasting"]
            plt.plot(x_for_n[:,0], x_for_n[:,1], color="green",label="n-forecasting")
        if lines["t-for"]:
            x_for_t = plots["trained-forecasting"]
            plt.plot(x_for_t[:,0], x_for_t[:,1], color="violet",label=f"{for_hor}-forecasting (trained)")
    except KeyError:
        return False

    plt.xlabel("x0(t)")
    plt.ylabel(f"x1(t)")
    plt.legend()
    plt.scatter(x_sys[Nt, 0], x_sys[Nt, 1], color="black") # training horizon position
    if show:
        plt.show()
    return True



def plot_all(components=3, plots=None, data=None, path=None):
    if plots is None:
        data, plots = _load(path)
    elif data is None:
        if path is None:
            data = save_handler.load_cfgs()
        else:
            data = save_handler.load_cfgs(path=path)
    
    # Loss
    res = plot_loss(plots, show=False)
    if res:
        ind = 2
    else:
        ind = 1


    # Err
    plt.figure(ind)
    try: 
        washout = data["washout"]
        threshold=data["threshold"]
        res = plot_err(plots, show=False, threshold=threshold, washout=washout)
    except:
        res = plot_err(plots, show=False)
    if res:
        ind += 1


    for k in range(components):
        plt.figure(ind)
        plot_component(k, plots, data, show=False)
        if res:
            ind += 1

    if components > 2:
        plt.figure(ind)
        plot3d(plots, data, lines={"gt":True, "1-for":False, "n-for":True, "t-for":True}, show=False)
        if res:
            ind += 1
    elif components ==2:
        plt.figure(ind)
        plot2d(plots, data, lines={"gt":True, "1-for":False, "n-for":True, "t-for":True}, show=False)
        if res:
            ind += 1

    plt.show()


if __name__ == "__main__":
    import data_save_handler as save_handler
    plot_all(path=os.path.abspath("..\data"))
else:
    from . import data_save_handler as save_handler


