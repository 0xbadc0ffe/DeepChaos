import os
import torch
import json
    

#PATH = os.path.abspath("..\data")
PATH = os.path.abspath(".\data")

def load(mod_name="model", jfile="cfgs", path=PATH):
    model_dict = None
    data = None
    plots = None
    try:
        data =  load_cfgs(jfile, path)
        model_dict = load_model_dict(mod_name,path)
        plots = load_plot_data(path)
    except Exception as e:
        #print(e)
        return False, model_dict, data, plots
    return True, model_dict, data, plots

def load_plot_data(path=PATH):
    plots = {}
    plots["loss"] = torch.load(path + "\\loss.pt")
    plots["weigths_norm"] = torch.load(path + "\\weigths_norm.pt")
    plots["ground_truth_sys"] = torch.load(path + "\\ground_truth_sys.pt")
    plots["1-forecasting"] = torch.load(path + "\\1-forecasting.pt")
    plots["n-forecasting"] = torch.load(path + "\\n-forecasting.pt")
    plots["trained-forecasting"] = torch.load(path + "\\trained-forecasting.pt")
    return plots

def load_cfgs(name="cfgs", path=PATH):
    with open(path +f"\{name}.json", "r") as jfile:
        return json.load(jfile)

def load_model_dict(name="model",path=PATH):
    return torch.load(path + f"\{name}.pth")

def save_hidden(h_0, name="h_0", path=PATH):
    torch.save(h_0, path + f"\{name}.pt")

def load_hidden(name="h_0", path=PATH):
    return torch.load(path + f"\{name}.pt")

def save_initial(x_0, name="x_0", path=PATH):
    torch.save(x_0, path + f"\{name}.pt")

def load_initial(name="x_0", path=PATH):
    return torch.load(path + f"\{name}.pt")

def save_W(W, name="W", path=PATH):
    torch.save(W, path + f"\{name}.pt")

def load_W(name="W", path=PATH):
    return torch.load(path + f"\{name}.pt")

def save_Win(Win, name="Win", path=PATH):
    torch.save(Win, path + f"\{name}.pt")

def load_Win(name="Win", path=PATH):
    return torch.load(path + f"\{name}.pt")

# TODO: Save System (and x_0 togheter)
# TODO: Save W_in, W, h_0 togheter

def save(model, data, plots: dict, model_name="model", cfgs_name="cfgs", path=PATH):

    # Saving model weights
    torch.save(model.state_dict(), path+f"\{model_name}.pth")

    # Saving plots and configs
    jsonfile = path+f"\{cfgs_name}.json"
    
    with open(jsonfile, "w+") as jfile:
        json.dump(data, jfile, indent=4)

    # saving plots
    for name in plots:
        torch.save(plots[name], path + f"\{name}.pt")
