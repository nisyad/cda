import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from utils import helper


def train_ganin(
    model,
    epoch,
    config,
    trainloader_src,
    trainloader_tgt,
    criterion_l,
    criterion_d,
    optimizer,
    device='cpu',
):

    model.train()
    running_loss_total = 0


    alpha = helper.get_alpha(epoch, config["epochs"])
    print("alpha: ", alpha)

    for batch_idx, ((imgs_src, lbls_src),
                    (imgs_tgt, _)) in enumerate(zip(trainloader_src,
                                                    trainloader_tgt),
                                                start=1):

        loss_total = 0

        optimizer.zero_grad()
        # source domain
        imgs_src, lbls_src = imgs_src.to(device), lbls_src.to(device)
        # imgs_src = torch.cat(3 * [imgs_src], 1)

        out_l, out_d = model(imgs_src, alpha)
        loss_l_src = criterion_l(out_l, lbls_src)
        actual_d = torch.zeros(out_d.shape).to(device)
        loss_d_src = criterion_d(out_d, actual_d)

        # target domain
        imgs_tgt = imgs_tgt.to(device)

        _, out_d = model(imgs_tgt, alpha)
        actual_d = torch.ones(out_d.shape).to(device)
        loss_d_tgt = criterion_d(out_d, actual_d)

        loss_total = loss_d_src + loss_l_src + loss_d_tgt
        loss_total.backward()
        optimizer.step()

        running_loss_total += loss_total

        if batch_idx % 30 == 0:
            print(
                f"Epoch: {epoch}/{config['epochs']} Batch: {batch_idx}/{len(trainloader_src)}"
            )
            print(f"Total Loss: {running_loss_total/batch_idx}")

    return running_loss_total / batch_idx



def test_ganin(model, testloader_tgt, device='cpu'):

    accuracy = 0
    model.eval()
    with torch.no_grad():
        
        for imgs, lbls in testloader_tgt:

            imgs, lbls = imgs.to(device), lbls.to(device)
            logits, _ = model(imgs, 0)  # alpha=0 for test
            # print("logits shape: ", logits.shape)

            # derive which class index corresponds to max value
            preds_l = torch.max(logits, dim=1)[1]
            # print("preds shape: ", preds_l.shape)
            # print("labels shape: ", lbls.shape)
            
            equals = torch.eq(preds_l,
                              lbls)  # count no. of correct class predictions
            accuracy += torch.mean(equals.float())

    print(f"Test accuracy: {accuracy / len(testloader_tgt)}")
    print("\n")

    return accuracy / len(testloader_tgt)
    


def train_simple_classifier(
    model,
    epoch,
    config,
    trainloader_src,
    criterion,
    optimizer,
    device='cpu',
):
    
    model.train()
    running_loss_total = 0


    alpha = helper.get_alpha(epoch, config["epochs"])
    print("alpha: ", alpha)

    for batch_idx, (imgs_src, lbls_src) in enumerate(trainloader_src, start=1):

        optimizer.zero_grad()

        imgs_src, lbls_src = imgs_src.to(device), lbls_src.to(device)

        out = model(imgs_src)

        loss = criterion(out, lbls_src)

        loss.backward()
        optimizer.step()

        running_loss_total += loss

        if batch_idx % 300 == 0:
            print(f"Epoch: {epoch}/{config['epochs']} Batch: {batch_idx}/{len(trainloader_src)}")
            print(f"Total Loss: {running_loss_total/batch_idx}")

    return running_loss_total / batch_idx



def test_simple_classifier(model, testloader_tgt, device='cpu'):

    accuracy = 0
    model.eval()
    with torch.no_grad():
        
        for imgs, lbls in testloader_tgt:

            imgs, lbls = imgs.to(device), lbls.to(device)
            logits = model(imgs) 
            # print("logits shape: ", logits.shape)

            # derive which class index corresponds to max value
            preds = torch.max(logits, dim=1)[1]
            # print("preds shape: ", preds.shape)
            # print("labels shape: ", lbls.shape)
            
            equals = torch.eq(preds,
                              lbls)  # count no. of correct class predictions
            accuracy += torch.mean(equals.float())

    print(f"Test accuracy: {accuracy / len(testloader_tgt)}")
    print("\n")

    return accuracy / len(testloader_tgt)
    





