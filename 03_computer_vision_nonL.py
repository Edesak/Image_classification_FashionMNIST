import torch
from torch.utils.data import DataLoader
import torchvision
import torchmetrics
from torch import nn
import matplotlib.pyplot as plt
import My_models as mm
import Helper_functions as hp
from tqdm import tqdm
import timeit
import pandas as pd
import random
from mlxtend.plotting import plot_confusion_matrix
from torchmetrics.classification import MulticlassConfusionMatrix
from pathlib import Path

#print(mlxtend.__version__)
BATCH_SIZE = 32
SHOW_RANDOM_IMG = False

device = hp.setup_device()

train_dataset = torchvision.datasets.FashionMNIST(root='data',
                                                     download=True,
                                                     transform=torchvision.transforms.ToTensor(),
                                                     target_transform=None,
                                                  train=True)

test_dataset = torchvision.datasets.FashionMNIST(root='data',
                                                 train=False,
                                                 download=True,
                                                 transform=torchvision.transforms.ToTensor(),
                                                 target_transform=None)

num_classes = len(train_dataset.classes)
print(f"Number of classes: {num_classes}")
class_names = train_dataset.classes
print(class_names)

print(f"\n {train_dataset} \n \n {test_dataset}")

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False)


batch_train_feature, batch_train_label = next(iter(train_dataloader))
batch_test_feature, batch_test_label = next(iter(test_dataloader))

print(f"\nTRAIN DATASET -> Batch size: {BATCH_SIZE} | shape of features: {batch_train_feature.shape} | shape of labels: {batch_train_label.shape} ")
print(f"TEST DATASET  -> Batch size: {BATCH_SIZE} | shape of features: {batch_test_feature.shape}  | shape of labels: {batch_test_label.shape} ")


if SHOW_RANDOM_IMG:
    hp.show_images_clas(num_rows=5,num_col=5,images=train_dataset.data,labels=train_dataset.targets,class_names=train_dataset.classes)

torch.manual_seed(42)
torch.cuda.manual_seed(42)
modelv2 = mm.Fashion_CNNv2(in_features=1,
                         hidden_units=10,
                         out_features=num_classes)

base_line = mm.baseline_model(input_shape=28*28,
                              hidden_units=10,
                              output_shape=num_classes)

modelv1 = mm.Fashion_modelv1(in_features=28*28,
                             hidden_units=10,
                             out_features=num_classes)

models = [modelv2]

def train_step_temp(model:nn.Module = None,
                    loss_fn:torch.nn.Module = None,
                    optimizer:torch.optim.Optimizer = None,
                    acc_fn:torchmetrics.Accuracy = None,
                    train_dataloader: DataLoader = None,
                    device:torch.device = None):
    model.to(device)
    acc_fn.to(device)
    model.train()
    total_loss,total_acc = 0,0

    for batch,(X,y) in enumerate(train_dataloader):

        X,y = X.to(device),y.to(device)

        train_logits = model(X)
        #Needs to update otherwise it will stack then call compute to get acc
        acc_fn.update(train_logits, y)
        total_acc += acc_fn.compute()

        loss = loss_fn(train_logits,y)
        total_loss += loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    total_loss = total_loss/len(train_dataloader)
    total_acc = total_acc*100/len(train_dataloader)

    return (total_loss,total_acc)

def val_step_temp(model:nn.Module = None,
                  val_dataloader:DataLoader = None,
                  loss_fn:torch.nn.Module = None,
                  acc_fn:torchmetrics.Accuracy = None,
                  device:torch.device = None):
    model.to(device)
    acc_fn.to(device)
    model.eval()
    total_loss, total_acc = 0, 0

    with torch.inference_mode():

        for batch,(X,y) in enumerate(val_dataloader):
            X,y = X.to(device), y.to(device)
            val_logits = model(X)

            acc_fn.update(val_logits,y)
            total_acc += acc_fn.compute()

            total_loss += loss_fn(val_logits,y)


        total_acc = total_acc*100/len(val_dataloader)
        total_loss = total_loss/len(val_dataloader)

    return (total_loss,total_acc)

def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               accuracy_fn: torchmetrics.Accuracy,
               device: torch.device = device):
    model.to(device)
    loss, acc = 0,0
    model.eval()
    with torch.inference_mode():
        for X,y in tqdm(data_loader):
            X,y = X.to(device),y.to(device)
            y_pred = model(X)

            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(y,y_pred.argmax(dim=1))

        loss /= len(data_loader)
        acc /= len(data_loader)

    return {"model name": model.__class__.__name__,
            "model_loss": loss.item(),
            "model_accuracy": acc.item()}

results = pd.DataFrame()

for model in models:

    loss_fn = torch.nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=model.parameters(),
                            lr=0.1)
    acc_fn = torchmetrics.Accuracy(task='multiclass',num_classes=num_classes)

    epochs = 3

    start = timeit.default_timer()


    pbar = tqdm(total=epochs,bar_format='{l_bar}{bar:50}{r_bar}', colour='green', leave=False)

    for epoch in range(epochs):
        train_loss, train_acc = train_step_temp(model=model,optimizer=optim,loss_fn=loss_fn,train_dataloader=train_dataloader,acc_fn=acc_fn,device=device)
        val_loss, val_acc = val_step_temp(model=model,loss_fn=loss_fn,acc_fn=acc_fn,val_dataloader=test_dataloader,device=device)

        pbar.update(1)
        tqdm.write("\033[34m" +f" | Train_loss: {train_loss:.4f} Train_acc: {train_acc:.2f}% | Val_loss: {val_loss:.4f} Val_acc: {val_acc:.2f}%" + "\033[0m")

        #print(f"\nEpoch: {epoch+1}/{epochs} | Train_loss: {train_loss:.4f} Train_acc: {train_acc:.2f}% | Val_loss: {val_loss:.4f} Val_acc: {val_acc:.2f}%")

    end = timeit.default_timer()
    hp.print_train_time(start=start,end=end)

    result = eval_model(model=model,
                        loss_fn=loss_fn,
                        accuracy_fn=acc_fn,
                        data_loader=test_dataloader,
                        device=device)
    print(result)
    result_df = pd.DataFrame.from_dict([result])
    results = pd.concat([results,result_df], ignore_index=True)

print(results)


def make_preds(model:torch.nn.Module,
               data:list,
               device:torch.device):

    pred_probs= []
    model.to('cpu')
    #data.to(device)
    model.eval()
    with torch.inference_mode():
        for sample in data:
            #ample.unsqueeze(sample,dim=0)s.to(device)
            sample.to('cpu')
            sample = sample.unsqueeze(dim=0)

            #print(sample)
            #print(model)
            pred_logits = model(sample)

            pred_prob = torch.softmax(pred_logits.squeeze(),dim=0)

            pred_probs.append(pred_prob.cpu())

    return torch.stack(pred_probs)

test_samples = []
test_labels = []
for sample,label in random.sample(list(test_dataset),k=9):
    test_samples.append(sample)
    test_labels.append(label)

pred_probs = make_preds(model=modelv2,
                        data=test_samples,
                        device=device)
pred_idx = torch.argmax(pred_probs,dim=1)

plt.figure(figsize=(9,9))
nrows=3
ncols=3
plt.axis(False)
for i,sample in enumerate(test_samples):
    plt.subplot(nrows,ncols,i+1)

    plt.imshow(sample.squeeze(), cmap="gray")

    pred_label = class_names[pred_idx[i]]

    truth_label = class_names[test_labels[i]]

    title_text = f"Pred: {pred_label} | Truth: {truth_label}"

    if pred_label == truth_label:
        plt.title(title_text,fontsize= 10,c='g')
    else:
        plt.title(title_text,fontsize =10, c='r')

plt.show()

y_preds = []
modelv2.to(device)
modelv2.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader,desc="Making predictions.."):
        X,y = X.to(device), y.to(device)

        y_logits = modelv2(X)
        y_pred = torch.softmax(y_logits.squeeze(),dim=0).argmax(dim=1)

        y_preds.append(y_pred.cpu())

    y_pred_tensor = torch.cat(y_preds)

cfmat = MulticlassConfusionMatrix(num_classes=num_classes)
cfmat_tensor = cfmat(preds=y_pred_tensor,
                     target=test_dataset.targets)
fig, ax = plot_confusion_matrix(conf_mat=cfmat_tensor.numpy(),
                                class_names=class_names,
                                figsize=(10,7))

plt.show()

MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True,
                 exist_ok=True)
MODEL_NAME = "pytorch_CV_CNN_modelv2.pth"

SAVE_PATH = MODEL_PATH/MODEL_NAME
torch.save(modelv2.state_dict(),
           f=SAVE_PATH)